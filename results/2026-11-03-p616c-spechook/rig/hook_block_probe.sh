SPEC_METHOD_FLAG=""
if [ "$ARC_SPEC_METHOD" = "probabilistic" ]; then
  SPEC_METHOD_FLAG=",\"draft_sample_method\":\"probabilistic\""
fi
SPEC_FLAG=""
if [ "$ARC_SPEC" != 0 ]; then
  SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$ARC_SPEC$SPEC_METHOD_FLAG}'"
fi
printf 'SPEC_FLAG=[%s]\n' "$SPEC_FLAG"
