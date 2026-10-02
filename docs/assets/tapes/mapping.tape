# cslcore map --test: bypass tricks against a hand-written mapper (still image)
Output docs/assets/mapping.gif
Source docs/assets/tapes/_settings.tape
Set Height 860

Hide
Type "export PS1='\[\e[1;36m\]$\[\e[0m\] ' PATH=$PWD/.venv/bin:$PATH && cd /tmp/csl-demo && clear"
Enter
Type "cd ws && clear"
Enter
Sleep 500ms
Type "clear; bash ../mapping/check.sh"
Enter
Sleep 4s
Show
Sleep 500ms
Screenshot docs/assets/mapping.png
Sleep 1s
