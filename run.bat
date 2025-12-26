@ECHO OFF

gcc %1.c utils.c mat.c mlp.c -o %1.exe -Wall -ggdb -lm 
.\%1.exe