"""Reject partial or failed batch output before recording integration success."""
from copy import deepcopy
from unittest import IsolatedAsyncioTestCase, TestCase

from scripts.verify_vertex_batch_integration import MARKER, validate_rows


class VertexBatchEvidenceTests(TestCase):
    def test_row_errors_cardinality_and_truncation_fail_closed(self):
        good = {"status": "", "response": {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": MARKER}]}}]}}
        validate_rows([good])
        for rows in ([], [good, good], [{**good, "status": "Bad Request"}], [{"response": {}}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_rows(rows)
        for reason in ("MAX_TOKENS", "SAFETY", None):
            bad = deepcopy(good)
            bad["response"]["candidates"][0]["finishReason"] = reason
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                validate_rows([bad])
        bad = deepcopy(good)
        bad["response"]["candidates"][0]["content"]["parts"] = [{"text": MARKER, "thought": True}, {"text": "different"}]
        with self.assertRaises(ValueError):
            validate_rows([bad])


class VertexBatchLifecycleTests(IsolatedAsyncioTestCase):
    async def test_resume_timeout_never_creates_or_deletes_resources(self):
        from argparse import Namespace
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import AsyncMock, Mock, patch
        from scripts import verify_vertex_batch_integration as runner
        from scripts import verify_vertex_integration as verifier

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / 'sdk.whl', project='p', model='gemini-3.8-flash', state=root / 'state.json', output=root / 'report.json')
            state = {'identity': {'project': 'p', 'model': args.model, 'location': 'global', 'wheel_sha256': 'a' * 64}, 'bucket': 'owned-test', 'bucket_created': True, 'job': 'projects/p/locations/global/batchPredictionJobs/existing'}
            args.state.write_text(json.dumps(state))
            client = Mock()
            client.create = AsyncMock()
            client.wait = AsyncMock(side_effect=TimeoutError)
            session = Mock()
            with patch.dict('sys.modules', {'verify_vertex_integration': verifier}), patch.object(verifier, 'verify_wheel', return_value='a' * 64), patch.object(runner.google.auth, 'default', return_value=(Mock(), 'p')), patch.object(runner, 'AuthorizedSession', return_value=session), patch.object(runner, 'create_vertex', return_value=Mock(batches=Mock(return_value=client))):
                self.assertEqual(await runner.run(args), 1)
            client.create.assert_not_called()
            client.wait.assert_awaited_once_with(state['job'], poll_interval_ms=10000, timeout_ms=120000)
            session.request.assert_not_called()
            self.assertEqual(json.loads(args.state.read_text()), state)
            self.assertEqual(json.loads(args.output.read_text())['operations']['poll'], 'pending')

    async def test_uncertain_creation_stops_before_authentication_or_submission(self):
        from argparse import Namespace
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from scripts import verify_vertex_batch_integration as runner
        from scripts import verify_vertex_integration as verifier

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / 'sdk.whl', project='p', model='gemini-3.8-flash', state=root / 'state.json', output=root / 'report.json')
            args.state.write_text(json.dumps({'identity': {'project': 'p', 'model': args.model, 'location': 'global', 'wheel_sha256': 'a' * 64}, 'bucket': 'owned-test', 'phase': 'job-dispatch'}))
            with patch.dict('sys.modules', {'verify_vertex_integration': verifier}), patch.object(verifier, 'verify_wheel', return_value='a' * 64), patch.object(runner.google.auth, 'default') as auth:
                with self.assertRaisesRegex(SystemExit, 'reconcile'):
                    await runner.run(args)
            auth.assert_not_called()

    async def test_success_validates_result_and_deletes_only_owned_generations(self):
        from argparse import Namespace
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import AsyncMock, Mock, patch
        from scripts import verify_vertex_batch_integration as runner
        from scripts import verify_vertex_integration as verifier

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / 'sdk.whl', project='p', model='gemini-3.8-flash', state=root / 'state.json', output=root / 'report.json')
            state = {'identity': {'project': 'p', 'model': args.model, 'location': 'global', 'wheel_sha256': 'a' * 64}, 'bucket': 'owned-test', 'bucket_created': True, 'job': 'projects/p/locations/global/batchPredictionJobs/existing'}
            args.state.write_text(json.dumps(state))
            client = Mock(create=AsyncMock(), wait=AsyncMock(return_value={'state': 'JOB_STATE_SUCCEEDED'}))
            row = {'status': '', 'response': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': MARKER}]}}]}}
            session = Mock()
            def request(method, url, **kwargs):
                response = Mock(text=json.dumps(row))
                response.json.return_value = {'items': [{'name': 'input.jsonl', 'generation': '11'}, {'name': 'output/predictions.jsonl', 'generation': '12'}]}
                return response
            session.request.side_effect = request
            with patch.dict('sys.modules', {'verify_vertex_integration': verifier}), patch.object(verifier, 'verify_wheel', return_value='a' * 64), patch.object(runner.google.auth, 'default', return_value=(Mock(), 'p')), patch.object(runner, 'AuthorizedSession', return_value=session), patch.object(runner, 'create_vertex', return_value=Mock(batches=Mock(return_value=client))):
                self.assertEqual(await runner.run(args), 0)
            client.create.assert_not_called()
            deletes = [call for call in session.request.call_args_list if call.args[0] == 'DELETE']
            self.assertEqual(len(deletes), 3)
            self.assertEqual(deletes[0].kwargs['params'], {'ifGenerationMatch': '11'})
            self.assertEqual(deletes[1].kwargs['params'], {'ifGenerationMatch': '12'})
            self.assertEqual(deletes[2].args[1], 'https://storage.googleapis.com/storage/v1/b/owned-test')
            self.assertTrue(json.loads(args.state.read_text())['cleaned'])
            self.assertEqual(json.loads(args.output.read_text())['operations']['results'], 'passed')
