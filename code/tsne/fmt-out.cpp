#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100

typedef float real;                    // Precision of float numbers

char vector_file[MAX_STRING], label_file[MAX_STRING], data_file[MAX_STRING];
int vertex_size, vector_size;

void TrainModel()
{
    FILE *fil, *fiv, *fo;
    double f_num;
    char word[MAX_STRING];
    
    fiv = fopen(vector_file, "rb");
    fil = fopen(label_file, "rb");
    fo = fopen(data_file, "wb");
    
    fread(&vertex_size, sizeof(int), 1, fiv);
    fread(&vector_size, sizeof(int), 1, fiv);
    
    for (int k = 0; k != vertex_size; k++)
    {
        fscanf(fil, "%s", word);
        fprintf(fo, "%s", word);
        for (int c = 0; c != vector_size; c++)
        {
            fread(&f_num, sizeof(double), 1, fiv);
            fprintf(fo, "\t%lf", f_num);
        }
        fprintf(fo, "\n");
    }
    
    printf("Vertex size: %d\n", vertex_size);
    printf("Vector size: %d\n", vector_size);
    
    fclose(fiv);
    fclose(fil);
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if ((i = ArgPos((char *)"-vector", argc, argv)) > 0) strcpy(vector_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-label", argc, argv)) > 0) strcpy(label_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data_file, argv[i + 1]);
    TrainModel();
    return 0;
}