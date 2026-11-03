import copy
import unittest

from gtm_agent import data_service
from gtm_agent.gtm_records import PROSPECTS


class UpdateProspectInfoTest(unittest.TestCase):
    def setUp(self):
        self._prospects = copy.deepcopy(PROSPECTS)
        self._profiles = copy.deepcopy(data_service._PROFILES)

    def tearDown(self):
        PROSPECTS.clear()
        PROSPECTS.update(self._prospects)
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self._profiles)

    def test_added_technology_is_persisted(self):
        result = data_service.update_prospect_info("LEAD-71001", "Kafka")

        self.assertTrue(result["updated"])
        self.assertIn("Kafka", result["tech_stack"])
        self.assertIn("Kafka", data_service.fetch_tech_stack("LEAD-71001"))

    def test_update_does_not_leave_stale_cached_profile(self):
        stale = {"prospect_id": "LEAD-71001", "tech_stack": list(PROSPECTS["LEAD-71001"]["tech_stack"])}
        data_service.save_profile_to_db("LEAD-71001", stale)

        data_service.update_prospect_info("LEAD-71001", "Kafka")

        profile = data_service.get_profile_from_db("LEAD-71001")["prospect_profile"]
        if profile is not None:
            self.assertIn("Kafka", profile["tech_stack"])

    def test_unknown_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("LEAD-00000", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})


if __name__ == "__main__":
    unittest.main()
