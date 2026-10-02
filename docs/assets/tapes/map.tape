# cslcore venom map: select agents, the 3D globe, dive into an agent
Output docs/assets/map.gif
Source docs/assets/tapes/_settings.tape
Set Width 1300
Set Height 840
Set Framerate 24

Hide
Type "export PS1='\[\e[1;36m\]$\[\e[0m\] ' PATH=$PWD/.venv/bin:$PATH && cd /tmp/csl-demo && clear"
Enter
Type "cd ws && clear"
Enter
Sleep 500ms
Show

Type "cslcore venom map"
Sleep 400ms
Enter
Sleep 4s
Type "2"
Sleep 1.5s
Type "s"
Sleep 2.5s
Type "1"
Sleep 2s
Type "2"
Sleep 2s
Enter
Sleep 3s
Escape
Sleep 1.5s
Hide
Type "q"
Sleep 500ms
