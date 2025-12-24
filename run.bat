@ECHO OFF

gcc %1.c utils.c mat.c -o %1.exe -Wall -Werror -ggdb -lm 
.\%1.exe