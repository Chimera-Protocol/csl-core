# cslcore watch: the live management panel while demo traffic flows
Output docs/assets/watch.gif
Source docs/assets/tapes/_settings.tape

Hide
Type "export PS1='\[\e[1;36m\]$\[\e[0m\] ' PATH=$PWD/.venv/bin:$PATH && cd /tmp/csl-demo && clear"
Enter
Type "(python $OLDPWD/scripts/venom_demo_traffic.py --workspace /tmp/csl-demo/ws --rate 25 >/dev/null 2>&1 &) && cd ws && clear"
Enter
Sleep 1s
Show

Type "cslcore watch"
Sleep 400ms
Enter
Sleep 5s
Down
Sleep 1s
Down
Sleep 1s
Enter
Sleep 3s
Escape
Sleep 1s
Tab
Sleep 3s
Escape
Sleep 1s
Type "g"
Sleep 5s
Down
Sleep 2.5s
Down
Sleep 2.5s
Hide
Type "q"
Sleep 1s
Type "pkill -f venom_demo_traffic; clear"
Enter
