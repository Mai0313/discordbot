# Temporary probe for #472

This file exists to prove in real CI that `--exclude-detectors=lob` reaches the
TruffleHog CLI and suppresses the finding. It carries the identifier that failed
`Secret Scanning` on #470:

`test_a_fetched_table_is_mirrored_to_disk`

If `Secret Scanning` passes with this file in the diff, the flag works. This file
is removed in the next commit and squashed away at merge.
