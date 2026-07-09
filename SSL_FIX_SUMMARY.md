# SSL Certificate Fix for Fireworks AI Integration

## Problem Diagnosis

The error log shows:
```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] 
  certificate verify failed: unable to get local issuer certificate (_ssl.c:1081)
```

### Root Cause
- macOS system Python installations do not include CA certificate bundles by default
- When `urllib.request.urlopen()` tries to make an HTTPS call to `https://api.fireworks.ai`,
  Python's SSL module cannot verify the server's certificate
- This causes all Fireworks AI API calls to fail

### Impact
- Image uploads complete successfully (UI works)
- PNG conversion succeeds
- Fireworks AI API call fails with SSL error
- No UIR/UMR files are written
- `/api/result/` and `/api/umr/` return 500 errors trying to serve non-existent files

## Solution

Updated `src/uir_pipeline/fireworks_vision.py` to use `certifi`'s CA certificate bundle:

```python
try:
    import certifi
    _ssl_context = _ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _ssl_context = _ssl.create_default_context()
```

Then pass the context to urllib:
```python
with _request.urlopen(req, timeout=120, context=_ssl_context) as resp:
    response_data = _json.loads(resp.read().decode("utf-8"))
```

### Why This Works
- `certifi` is a curated CA bundle that's already installed in your `.venv`
- It provides the necessary root certificates for HTTPS verification
- The SSL context is created once per API call and used for the HTTPS connection
- Fallback to default context if certifi is not available

## Verification

After the fix:
```bash
# Test SSL context creation
$ .venv/bin/python3 -c "import ssl, certifi; print(ssl.create_default_context(cafile=certifi.where()))"
✓ SSL context created with certifi CA bundle

# Test HTTPS connection
$ .venv/bin/python3 -c "import ssl, certifi, urllib.request; ..."
Connection test: HTTPError: HTTP Error 403: Forbidden
# ✓ This is expected - we got a real HTTP response, not an SSL error!
```

## Additional Notes

### Secondary Issue (Missing UIR File)
The web layer should handle API failures more gracefully. When the Fireworks call fails:
- The error is logged correctly
- But the runner still tries to serve files that were never written

This is partially handled by `fail-soft` error dicts, but the web layer could be 
improved to show the error in the UI instead of a 500 error.

### Environment Setup
If you encounter SSL issues in other Python projects on macOS, you can also:

```bash
# Run the macOS Python certificates install script
/Applications/Python 3.x/Install Certificates.command

# Or install certifi globally
pip install certifi
```

## Test Results

All tests pass after the fix:
```
============================= test session starts ==============================
collected 30 items

tests/test_fireworks_vision.py::TestImageConversion::test_png_passthrough PASSED
tests/test_fireworks_vision.py::TestImageConversion::test_jpeg_to_png PASSED
...

tests/test_fireworks_vision.py::TestRunImagePipeline::test_dry_run_full_flow PASSED

============================== 30 passed in 0.91s ==============================
```

## Next Steps

1. Try uploading an image again through the web UI
2. The SSL error should be resolved
3. Fireworks AI API calls should succeed (assuming valid API key)
4. UIR and UMR files will be created and served correctly

If you still see errors, check:
- `FIREWORKS_API_KEY` is set correctly in `.env`
- Fireworks AI service is reachable from your network
- Your API key has vision model access
