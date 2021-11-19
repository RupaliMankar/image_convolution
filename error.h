#include <stdio.h>
#include <iostream>
using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

#ifndef CUDA_HANDLE_ERROR_H
#define CUDA_HANDLE_ERROR_H

//handle error macro
static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
			//FILE* outfile = fopen("cudaErrorLog.txt", "w");
      		//fprintf(outfile,  "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
			//fclose(outfile);
            printf("%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
       		//exit( EXIT_FAILURE );

   	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void CufftError( cufftResult err )
{
    if (err != CUFFT_SUCCESS)
    {
        if(err == CUFFT_INVALID_PLAN)
            cout<<"The plan parameter is not a valid handle."<<endl;
        else if(err == CUFFT_ALLOC_FAILED)
            cout<<"Allocation failed."<<endl;
        else if(err == CUFFT_INVALID_VALUE)
            cout<<"At least one of the parameters idata, odata, and direction is not valid."<<endl;
        else if(err == CUFFT_INTERNAL_ERROR)
            cout<<"An internal driver error was detected."<<endl;
        else if(err == CUFFT_EXEC_FAILED)
            cout<<"CUFFT failed to execute the transform on the GPU."<<endl;
        else if(err == CUFFT_SETUP_FAILED)
            cout<<"The CUFFT library failed to initialize."<<endl;
        else
            cout<<"Unknown error: "<<err<<endl;

    }
}



#endif
