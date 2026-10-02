# cslcore setup: the guided flow, accepting the recommended choice at each step
Output docs/assets/setup.gif
Source docs/assets/tapes/_settings.tape
Set PlaybackSpeed 1.8
Set Framerate 12

Hide
Type "export PS1='\[\e[1;36m\]$\[\e[0m\] ' PATH=$PWD/.venv/bin:$PATH && cd /tmp/csl-demo && clear"
Enter
Type "rm -rf .csl policies && clear"
Enter
Sleep 500ms
Show

Type "cslcore setup --root host"
Sleep 400ms
Enter
Sleep 2.5s
# start, then accept each step
Enter
Sleep 7s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 2.5s
Enter
Sleep 2.5s
Enter
Sleep 3s
Enter
Sleep 3s
Enter
Sleep 5s
