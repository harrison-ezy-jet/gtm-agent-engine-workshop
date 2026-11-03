import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import send_prospect_email

FROM_REP = {"name": "Test Rep", "email": "rep@example.com"}


def _send(prospect):
    return send_prospect_email.func(
        prospect=prospect, subject="Hello", body="Hi there", runtime=None, from_rep=FROM_REP
    )


def _contact(prospect_id):
    record = data_service.get_prospect_record(prospect_id)
    return {"prospect_id": prospect_id, "name": record["name"], "email": record["email"]}


class SendProspectEmailTest(unittest.TestCase):
    def test_disqualified_source_record_blocks_send(self):
        result = _send(_contact("LEAD-50001"))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "disqualified")
        self.assertNotIn("message_id", result)

    def test_disqualified_flag_on_passed_prospect_blocks_send(self):
        prospect = {**_contact("LEAD-12853"), "disqualified": True}
        result = _send(prospect)
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("message_id", result)

    def test_disqualified_flag_without_source_record_blocks_send(self):
        result = _send({"name": "Unknown", "email": "unknown@example.com", "disqualified": True})
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("message_id", result)

    def test_qualified_prospect_is_sent(self):
        prospect = _contact("LEAD-12853")
        result = _send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])
        self.assertIn("message_id", result)


if __name__ == "__main__":
    unittest.main()
