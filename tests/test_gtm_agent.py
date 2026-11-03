import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.messages import AIMessage

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_agent import ProspectScore, RubricBreakdown, score_prospect, send_prospect_email


def _runtime(user_id):
    return SimpleNamespace(config={"metadata": {"user_id": user_id}})


def _fake_score():
    return ProspectScore(
        score=42,
        justification="ok",
        rubric_breakdown=RubricBreakdown(revenue_fit=40, tech_stack_match=40, segment_fit=40),
    )


class SendProspectEmailTest(unittest.TestCase):
    def test_resolves_prospect_by_id_and_defaults_sender(self):
        prospect = data_service.get_prospect_record("LEAD-50005")
        rep = data_service.get_rep("rep_rgarcia")
        result = send_prospect_email.func(
            prospect_id="LEAD-50005", subject="Hello", body="Body", runtime=_runtime("rep_rgarcia"),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["to_name"], prospect["name"])
        self.assertEqual(result["from"], rep["email"])
        self.assertEqual(result["from_name"], rep["name"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Body")

    def test_missing_prospect_returns_failed_result(self):
        result = send_prospect_email.func(
            prospect_id="LEAD-00000", subject="Hello", body="Body", runtime=_runtime("rep_rgarcia"),
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)


class ScoreProspectTest(unittest.TestCase):
    def setUp(self):
        self.llm = mock.MagicMock()
        self.llm.invoke.return_value = _fake_score()
        patcher = mock.patch.object(agent_module, "_scoring_llm", self.llm)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _scoring_input(self):
        return self.llm.invoke.call_args.args[0][1]["content"]

    def test_scores_from_ids_without_prefetched_records(self):
        result = score_prospect.invoke({"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"})
        self.assertEqual(result["score"], 42)
        content = self._scoring_input()
        self.assertIn("OFFER-10005", content)
        self.assertIn("LEAD-71001", content)

    def test_missing_prospect_returns_error(self):
        result = score_prospect.invoke({"prospect_id": "LEAD-00000", "offering_id": "OFFER-10005"})
        self.assertIsNone(result["score"])
        self.assertIn("error", result)
        self.llm.invoke.assert_not_called()

    def test_missing_offering_returns_error(self):
        result = score_prospect.invoke({"prospect_id": "LEAD-71001", "offering_id": "OFFER-00000"})
        self.assertIsNone(result["score"])
        self.assertIn("error", result)
        self.llm.invoke.assert_not_called()

    def test_scores_against_tech_stack_of_record(self):
        with mock.patch.object(data_service, "fetch_tech_stack", return_value=["TechOfRecord"]):
            score_prospect.invoke({"prospect_id": "LEAD-39002", "offering_id": "OFFER-10004"})
        self.assertIn("TechOfRecord", self._scoring_input())


class RunAgentTest(unittest.TestCase):
    def test_passes_signed_in_rep_and_keeps_rep_tool(self):
        rep = data_service.get_rep("rep_amills")
        fake = mock.MagicMock()
        fake.invoke.return_value = {"messages": [AIMessage("done")]}
        with mock.patch.object(agent_module, "gtm_agent", fake):
            result = agent_module.run_agent("Email LEAD-12853", user_id="rep_amills")
        self.assertEqual(result["reply"], "done")
        messages = fake.invoke.call_args.args[0]["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Email LEAD-12853"})
        self.assertTrue(any(rep["name"] in m["content"] and m["role"] == "system" for m in messages))
        tools = agent_module.gtm_agent.nodes["tools"].bound.tools_by_name
        self.assertIn("get_current_rep", tools)


if __name__ == "__main__":
    unittest.main()
