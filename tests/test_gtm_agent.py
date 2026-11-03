import json
import os
import unittest
from unittest import mock

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service, gtm_agent
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS

PROSPECT_ID = "LEAD-12853"
SENSITIVE_KEYS = ("billing_qualification", "card_on_file", "tax_id", "date_of_birth", "credit_check_ref")


class ProspectPiiTest(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.pop(PROSPECT_ID, None)
        self.sensitive_values = [str(v) for v in PROSPECTS[PROSPECT_ID]["billing_qualification"].values()]

    def tearDown(self):
        data_service._PROFILES.pop(PROSPECT_ID, None)

    def assertNoSensitiveData(self, serialized):
        for needle in (*SENSITIVE_KEYS, *self.sensitive_values):
            self.assertNotIn(needle, serialized)

    def test_get_prospect_returns_only_contact_fields(self):
        result = gtm_agent.get_prospect.invoke({"prospect_id": PROSPECT_ID})
        record = PROSPECTS[PROSPECT_ID]
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect"], {
            "prospect_id": PROSPECT_ID,
            "name": record["name"],
            "email": record["email"],
            "disqualified": record["disqualified"],
            "annual_revenue": record["annual_revenue"],
            "enrichment_source": record["enrichment_source"],
        })
        self.assertNoSensitiveData(json.dumps(result))

    def test_build_prospect_profile_omits_billing_qualification(self):
        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})
        profile = result["prospect_profile"]
        self.assertTrue(result["found"])
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(profile["tech_stack"], PROSPECTS[PROSPECT_ID]["tech_stack"])
        self.assertNoSensitiveData(json.dumps(result))
        self.assertNoSensitiveData(json.dumps(data_service._PROFILES[PROSPECT_ID]))

    def test_build_prospect_profile_filters_cached_profile(self):
        data_service.save_profile_to_db(PROSPECT_ID, {"prospect_id": PROSPECT_ID, **PROSPECTS[PROSPECT_ID]})
        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})
        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(profile["name"], PROSPECTS[PROSPECT_ID]["name"])
        self.assertNoSensitiveData(json.dumps(result))

    def test_score_prospect_sends_only_scoring_fields(self):
        record = PROSPECTS[PROSPECT_ID]
        profile = {"prospect_id": PROSPECT_ID, **record, "notes": "unrelated"}
        offering = OFFERINGS["OFFER-10001"]
        fake_llm = mock.Mock()
        fake_llm.invoke.return_value = gtm_agent.ProspectScore(
            score=50,
            justification="ok",
            rubric_breakdown=gtm_agent.RubricBreakdown(revenue_fit=50, tech_stack_match=50, segment_fit=50),
        )
        with mock.patch.object(gtm_agent, "_scoring_llm", fake_llm):
            result = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        self.assertEqual(result["score"], 50)
        user_content = fake_llm.invoke.call_args.args[0][1]["content"]
        sent_profile = json.loads(user_content.split("Prospect profile:\n", 1)[1])
        self.assertEqual(sent_profile, {
            "prospect_id": PROSPECT_ID,
            "name": record["name"],
            "annual_revenue": record["annual_revenue"],
            "tech_stack": record["tech_stack"],
            "account_details": record["account_details"],
            "engagement_history": record["engagement_history"],
        })
        self.assertNoSensitiveData(user_content)


if __name__ == "__main__":
    unittest.main()
