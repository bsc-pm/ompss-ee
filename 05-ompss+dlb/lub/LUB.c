#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include <mpi.h>

/*
Version que la matriz se guarda con punteros a las filas
Bien para enviar por filas.
*/


int howmany;
int whoAmI;

int N;
int B;
int NB;

#ifdef CHECK_RESULT
double **A_orig;
#endif

#define MIN(x,y)  ((x) < (y) ? (x) : (y))

double **A;

int block_owner(int block){
    /* NB=10 howmany=4
     * blocks:      |0|1|2|3|4|5|6|7|8|9|
     * owner:        0 0 0 1 1 1 2 2 3 3
     * blocksXproc:      C+1    -   C
     * L:                       ^
     */

    int owner;
    int C = NB/howmany;
    int L = NB%howmany * (C+1);

    if (block >= L){
        owner = NB%howmany + ((block-L)/C);
    }else{
        owner= block/(C+1);
    }
    return owner;
}


int translate (int i){
    int block, processor, local_i, C, L;
    block=i/B;
    C=NB/howmany; //How many blocks per proc at least
    L=(NB)%howmany; //How many procs have C+1 blocks

    processor=block_owner(block);
    if(processor==whoAmI){
        if (whoAmI<L){
            local_i = i - ( (whoAmI*(C+1)) *B);
        }else{
            local_i = i - ( ((L*(C+1)) + ((whoAmI-L) * C) )*B);
        }
    }else{
        local_i=-1;
    }
    return local_i;
}


void genmat (int master){
    int init_val, i, j, diag_dist;
    init_val = 1325;
    int local_i;

    for (i = 0; i < N; i++){
        local_i=translate(i);
        for (j = 0; j < N; j++) {
            init_val = (3125 * init_val) % 65536;
            diag_dist = (i<j) ? j-i: i-j;
            if (local_i!=-1) A[local_i][j] = 1000.0/(diag_dist+1) + (init_val-32768.0)/16384.0;
#ifdef CHECK_RESULT
            if (whoAmI==0)
                A_orig[i][j] = 1000.0/(diag_dist+1) + (init_val-32768.0)/16384.0;
#endif
        }
    }
}

#ifdef _OMPSS
#pragma omp task inout([B]k_block)
#endif
void lu0(int kk, double k_block[B][N]) {
    int i, j, k;
    int kinf, ksup;

    kinf = 0;
    ksup = B;

    if (((kk+1)*B)>N) ksup=N%B;

    for (k=kinf; k<ksup; k++) {
        for (i=k+1; i<ksup; i++) {
            k_block[i][k] = k_block[i][k] / k_block[k][k];
            for (j=k+1; j<ksup; j++)
                k_block[i][j] = k_block[i][j] - k_block[i][k] * k_block[k][j];
        }
    }
}

#ifdef _OMPSS
#pragma omp task in([B]k_block) inout([B]i_block)
#endif
void bdiv(int ii, int kk, double k_block[B][N], double i_block[B][N]) {
    int i, j, k;
    int kinf, ksup, iinf, isup;

    kinf = 0;
    ksup = B;
    if (((kk+1)*B)>N) ksup=N%B;

    iinf = 0;
    isup = B;
    if (((ii+1)*B)>N) isup=N%B;

    for (i=iinf; i<isup; i++) {
        for (k=kinf; k<ksup; k++) {
            i_block[i][k] = i_block[i][k] / k_block[k][k];
            for (j=k+1; j<ksup; j++){
                i_block[i][j] = i_block[i][j] - i_block[i][k]*k_block[k][j];
            }
        }
    }
}

#ifdef _OMPSS
#pragma omp task in([B]ik_block, [B]kj_block) inout([B]ij_block)
#endif
void bmod(int ii, int jj, int kk,
        double ik_block[B][N], double kj_block[B][N], double ij_block[B][N]) {
    int i, j, k;
    int kinf, ksup, iinf, isup, jinf, jsup;

    kinf = 0;
    ksup = B;
    if (((kk+1)*B)>N) ksup=N%B;

    iinf = 0;
    isup = B;
    if (((ii+1)*B)>N) isup=N%B;

    jinf = 0;
    jsup = B;
    if (((jj+1)*B)>N) jsup=N%B;

    for (i=iinf; i<isup; i++){
        for (k=kinf; k<ksup; k++){
            for (j=jinf; j<jsup; j++){
                ij_block[i][j] = ij_block[i][j] - ik_block[i][k]*kj_block[k][j];
            }
        }
    }
}

#ifdef _OMPSS
#pragma omp task in([B]k_block) inout([B]j_block)
#endif
void fwd(int kk, int jj, double k_block[B][N], double j_block[B][N]) {
    int i, j, k;
    int kinf, ksup, iinf, isup, jinf, jsup;

    kinf = 0;
    ksup = B;
    if (((kk+1)*B)>N) ksup=N%B;

    jinf = 0;
    jsup = B;
    if (((jj+1)*B)>N) jsup=N%B;

    for (k=kinf; k<ksup; k++) {
        for (i=k+1; i<ksup; i++) {
            for (j=jinf; j<jsup; j++)
                j_block[i][j] = j_block[i][j] - k_block[i][k]*j_block[k][j];
        }
    }
}

long usecs (void) {
    struct timeval t;

    gettimeofday(&t,NULL);
    return t.tv_sec*1000000+t.tv_usec;
}

#ifdef CHECK_RESULT
int check_result() {
    double L[N][N];
    double U [N][N];
    int i,j, local_i, k;
    double x;
    int processor;
    double * thisRow;
    MPI_Status status;
    for (i = 0; i < N; i++){
        processor=block_owner(i/B);
        if (processor==0){
            local_i=translate(i);
            thisRow=A[local_i];
        }else{
            MPI_Recv(thisRow, N, MPI_DOUBLE, processor, i, MPI_COMM_WORLD, &status);
        }
        for (j = 0; j < N; j++) {
#ifdef PRINT_RESULT
            printf("%f ", thisRow[j]);
#endif
            if (i==j)     { L[i][j]=1.0;      U[i][j]=thisRow[j]; }
            else if (i>j) { L[i][j]=thisRow[j];  U[i][j]=0.0;}
            else           { L[i][j]=0.0;      U[i][j]=thisRow[j]; }
        }
#ifdef PRINT_RESULT
        printf("\n");
#endif
    }

    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++) {
            x=0;
            for (k = 0; k < N; k++) {
                x+=L[i][k]*U[k][j];
            }
            if(((x-A_orig[i][j])*(x-A_orig[i][j]))>0.0000001L){
                printf("Error: posicion %d %d LxU=%f A=%f\n", i, j, x, A_orig[i][j]);
                // printf("|X");
            } else {} //printf("| ");
        }
        //printf("|\n");
    }
}
#endif

int main(int argc, char* argv[]) {
    long t_start,t_end;
    double time;
    int myrows, nrows, block;
    int ii, jj, kk, i, j, r, t, x;
    int local_i, local_kk, isup, jsup;
    int root;
    MPI_Status status;
    double *block_buffer;
    double **buffer_rcv, **buffer_tmp;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &howmany);
    MPI_Comm_rank(MPI_COMM_WORLD, &whoAmI);

    if (argc<3){
        if (whoAmI==0) {
            printf("missing arguments: LUB size-matrix size-block\n"
                    "Using default values: size-matrix=10000 and size-block=200\n");
        }
        N=10000;
        B=200;
    }else{
        N=atoi(argv[1]);
        B=atoi(argv[2]);
        if (whoAmI==0) {
            printf("MatrixSize: %d\n BlockSize: %d\n", N, B);
        }
    }

    //Para redondear al alza, si N=11 y B=5 entonces NB=2 y se queda un bloque sin asignar
    //de esta manera NB=3
    NB = (N+B-1)/B;

    myrows=NB*B;
    A = malloc(myrows*sizeof(double *));

    //Reserva bloques de tamaño N*B para que los elementos esten consecutivos e
    //memoria para poder enviarlo con una sola llamada MPI. Cada posicion de A
    //apunta a una fila de N elementos
    ii=0;
    for (block=0; block<NB; block++){
        block_buffer =malloc(N*B*sizeof(double));
        for (nrows=0; nrows<B; nrows++) {
            A[ii++] = block_buffer;
            block_buffer+=N;
        }
    }

#ifdef CHECK_RESULT
    if (whoAmI==0){
        A_orig = malloc(N*sizeof(double*));
        for (nrows=0; nrows<N; nrows++) {
            block_buffer =malloc(N*sizeof(double));
            A_orig[nrows]=block_buffer;
        }
    }
#endif
    genmat(whoAmI);

    buffer_rcv=malloc(B*sizeof(double *));
    ii=0;
    block_buffer =malloc(B*N*sizeof(double));
    for (nrows=0; nrows<B; nrows++) {
        buffer_rcv[ii++] = block_buffer;
        block_buffer+=N;
    }

    buffer_tmp=malloc(B*sizeof(double *));
    for (r=0; r<B; r++) buffer_tmp[r]=buffer_rcv[r];

    MPI_Barrier(MPI_COMM_WORLD);
    t_start=usecs();

    typedef double (*BLOCK)[N];

    for (kk=0; kk<NB; kk++) {
        if (block_owner(kk)==whoAmI) {
            local_kk=translate(kk*B);

            lu0(kk, &A[local_kk][kk*B]);

            for (jj=kk+1; jj<NB; jj++) {

                fwd(kk, jj,  &(A[local_kk][kk*B]),  &(A[local_kk][jj*B]));

            }

#ifdef _OMPSS
#pragma omp taskwait
#endif

            local_i=translate(kk*B);

            MPI_Bcast(A[local_i], N*B, MPI_DOUBLE, whoAmI, MPI_COMM_WORLD);

            for(r=0; r<B; r++) buffer_rcv[r]=A[r+local_i];

        }else{
#ifdef _OMPSS
#pragma omp taskwait
#endif
            root = block_owner(kk);

            MPI_Bcast(buffer_rcv[0], N*B, MPI_DOUBLE, root, MPI_COMM_WORLD);
        }

        for (ii=kk+1; ii<NB; ii++) {
            if(block_owner(ii)==whoAmI){
                local_i=translate(ii*B);

                bdiv(ii, kk, &(buffer_rcv[0][kk*B]), &(A[local_i][kk*B]));

                for (jj=kk+1; jj<NB; jj++){

                    bmod(ii, jj, kk, &(A[local_i][kk*B]),
                            &(buffer_rcv[0][jj*B]), &(A[local_i][jj*B]));

                }
            }
        }
        for (r=0; r<B; r++) buffer_rcv[r]= buffer_tmp[r];
    }


    MPI_Barrier(MPI_COMM_WORLD);
    t_end=usecs();


    time = ((double)(t_end-t_start))/1000000;

    MPI_Barrier(MPI_COMM_WORLD);
#ifdef CHECK_RESULT
    if (whoAmI==0){
        check_result();
    }else{
        for(i=0; i<N; i++){
            if((local_i=translate(i)) != -1){
                MPI_Send(A[local_i], N, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
            }
        }
    }
#endif

    if (whoAmI ==0) printf("\nprocess %d, time to compute = %f\n\n", whoAmI, time);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
