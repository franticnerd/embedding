#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100

typedef float real;                    // Precision of float numbers

char vector_file[MAX_STRING], label_file[MAX_STRING], data_file[MAX_STRING];
int vertex_size, vector_size, no_dim = 2, nshows = 0;
double ppl = 30, theta = 0.5;
double *vec;

void TrainModel()
{
    FILE *fi, *fol, *fod;
    char ch, word[MAX_STRING];
    double f_num;
    
    fi = fopen(vector_file, "rb");
    if (fi == NULL) {
        printf("Vector file not found\n");
        exit(1);
    }
    fol = fopen(label_file, "wb");
    fod = fopen(data_file, "wb");
    
    fscanf(fi, "%d %d", &vertex_size, &vector_size);
    if (nshows == 0) nshows = vertex_size;
    
    vec = (double *)malloc(vertex_size * vector_size * sizeof(double));
    
    for (int k = 0; k != vertex_size; k++)
    {
        fscanf(fi, "%s", word);
        ch = fgetc(fi);
        if (k < nshows) fprintf(fol, "%s\n", word);
        for (int c = 0; c != vector_size; c++)
        {
            //fread(&f_num, sizeof(real), 1, fi);
            fscanf(fi, "%lf", &f_num);
            vec[k * vector_size + c] = (double)f_num;
        }
    }
    
    printf("Vertex size: %d\n", vertex_size);
    printf("Vector size: %d\n", vector_size);
    printf("Nshows: %d\n", nshows);
    printf("Ndims: %d\n", no_dim);
    printf("Theta: %lf\n", theta);
    printf("PPL: %lf\n", ppl);
    
    fwrite(&nshows, sizeof(int), 1, fod);
    fwrite(&vector_size, sizeof(int), 1, fod);
    fwrite(&theta, sizeof(double), 1, fod);
    fwrite(&ppl, sizeof(double), 1, fod);
    fwrite(&no_dim, sizeof(int), 1, fod);
    for (int k = 0; k != nshows; k++) for (int c = 0; c != vector_size; c++)
        fwrite(&(vec[k * vector_size + c]), sizeof(double), 1, fod);
    
    fclose(fi);
    fclose(fol);
    fclose(fod);
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
    if ((i = ArgPos((char *)"-nshows", argc, argv)) > 0) nshows = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-no-dim", argc, argv)) > 0) no_dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-theta", argc, argv)) > 0) theta = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-ppl", argc, argv)) > 0) ppl = atof(argv[i + 1]);
    TrainModel();
    return 0;
}