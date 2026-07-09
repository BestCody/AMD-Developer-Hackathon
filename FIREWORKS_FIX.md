# Fireworks AI Integration - Final Fix

## Problem Diagnosis

The error log showed: `HTTP 403: error code: 1010`

### Root Causes (Fixed)

1. **Wrong Model Name**: Used `qwen3-vl-8b-instruct` which returned 403
   - **Fixed**: Changed to `minimax-m3` (verified working model from example.tmp)

2. **Wrong HTTP Library**: Used `urllib` instead of `requests`
   - **Fixed**: Rewrote to use `requests` library (matching Fireworks' documented API)

## Changes Made

### File: `src/uir_pipeline/fireworks_vision.py`
- Line 45: `_DEFAULT_VISION_MODEL = "accounts/fireworks/models/minimax-m3"` (was qwen3-vl-8b-instruct)
- Lines 197-246: Replaced urllib with requests library
- Uses `requests.post()` with proper error handling
- Matches exact format from example.tmp

### File: `tests/test_fireworks_vision.py`
- Line 121: Updated test to expect `minimax` instead of `qwen3`

## Verification

All 50 tests pass:
- 30 fireworks_vision tests ✓
- 20 web tests ✓

## Why This Works

Based on the working example in example.tmp:
- Uses `minimax-m3` model (vision-capable model on Fireworks serverless)
- Uses `requests` library (simpler, better error handling)
- Uses OpenAI-compatible chat format with `image_url`

## Next Steps

1. Restart web server: `python web.py`
2. Upload an image through the UI
3. The 403 error should be resolved
4. Image will be analyzed and UIR/UMR generated

## Environment

The fix uses:
- Model: `accounts/fireworks/models/minimax-m3`
- Library: `requests` (already in requirements.txt)
- Format: OpenAI vision API with `image_url` content type