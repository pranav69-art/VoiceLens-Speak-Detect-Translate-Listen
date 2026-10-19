import io
import unittest
from unittest.mock import patch

import app as webapp


class VoiceLensWebTests(unittest.TestCase):
    def setUp(self):
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def test_home_serves_app(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"VoiceLens", response.data)
        response.close()

    def test_health_reports_limits_and_model(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["limits"]["maxAudioSeconds"], 180)
        self.assertIn("speechModel", payload)

    def test_transcribe_returns_model_result(self):
        expected = {"text": "hello world", "language": "en", "languageProbability": 0.99}
        with patch.object(webapp, "transcribe_audio", return_value=expected) as inference:
            response = self.client.post("/api/transcribe", data=b"recorded audio", content_type="audio/webm")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        inference.assert_called_once_with(b"recorded audio", "audio/webm")

    def test_rejects_empty_or_unsupported_audio(self):
        empty = self.client.post("/api/transcribe", data=b"", content_type="audio/webm")
        unsupported = self.client.post("/api/transcribe", data=b"data", content_type="text/plain")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(unsupported.status_code, 415)

    def test_reports_missing_model_dependencies(self):
        with patch.object(webapp, "transcribe_audio", side_effect=RuntimeError("Install the speech dependencies")):
            response = self.client.post("/api/transcribe", data=b"recorded audio", content_type="audio/webm")
        self.assertEqual(response.status_code, 503)
        self.assertIn("speech dependencies", response.get_json()["error"])

    def test_optional_deployment_token_protects_inference(self):
        expected = {"text": "hello", "language": "en"}
        with patch.object(webapp, "API_TOKEN", "test-token"), patch.object(webapp, "transcribe_audio", return_value=expected):
            denied = self.client.post("/api/transcribe", data=b"audio", content_type="audio/webm")
            allowed = self.client.post(
                "/api/transcribe", data=b"audio", content_type="audio/webm",
                headers={"Authorization": "Bearer test-token"},
            )
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_returns_json_for_oversized_upload(self):
        with patch.dict(webapp.app.config, {"MAX_CONTENT_LENGTH": 8}):
            response = self.client.post(
                "/api/transcribe", data=io.BytesIO(b"x" * 32), content_type="audio/webm"
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.mimetype, "application/json")


if __name__ == "__main__":
    unittest.main()
