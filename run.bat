@ECHO OFF

gcc %1.c mat.c utils.c metrics.c mlp.c -o %1.exe -Wall -ggdb -lm
.\%1.exe