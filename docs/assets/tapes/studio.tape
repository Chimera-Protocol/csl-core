# cslcore studio: Z3 (ctrl+r / F5) and TLA+ (ctrl+t / F8) on a live policy
Output docs/assets/studio.gif
Source docs/assets/tapes/_settings.tape

Hide
Type "export PS1='\[\e[1;36m\]$\[\e[0m\] ' PATH=$PWD/.venv/bin:$PATH && cd /tmp/csl-demo && clear"
Enter
Type "cd ws && clear"
Enter
Sleep 500ms
Show

Type "cslcore studio --agent ingest-worker"
Sleep 400ms
Enter
Sleep 3s
Ctrl+R
Sleep 6s
Ctrl+T
Sleep 13s
Hide
Ctrl+Q
Sleep 1s
