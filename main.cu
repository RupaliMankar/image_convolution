#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

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
            std::cout<<"The plan parameter is not a valid handle."<<std::endl;
        else if(err == CUFFT_ALLOC_FAILED)
            std::cout<<"Allocation failed."<<std::endl;
        else if(err == CUFFT_INVALID_VALUE)
            std::cout<<"At least one of the parameters idata, odata, and direction is not valid."<<std::endl;
        else if(err == CUFFT_INTERNAL_ERROR)
            std::cout<<"An internal driver error was detected."<<std::endl;
        else if(err == CUFFT_EXEC_FAILED)
            std::cout<<"CUFFT failed to execute the transform on the GPU."<<std::endl;
        else if(err == CUFFT_SETUP_FAILED)
            std::cout<<"The CUFFT library failed to initialize."<<std::endl;
        else
            std::cout<<"Unknown error: "<<err<<std::endl;

    }
}


const float pi = 3.14159;
size_t X, Y;
size_t k = 0;

float normgaussian (size_t x, float sigma) {
	return std::exp( -((x/(sigma))*(x/(sigma)))/2.0 );
}

void createKernel(float* gkernel, int sigma, size_t k){
	float constant = 1/(sqrt(2 * pi) * sigma);
	size_t offset = k/2;
	
	for(size_t x = 0; x< k; x++){
		gkernel[x] = constant * normgaussian((x-offset), sigma);
	}
	
}

unsigned char* readImg(std::string filename){
	std::ifstream infile;
    std::string data;
    infile.open(filename, std::ios::binary|std::ios::in);
    if(!infile.is_open()){
        std::cout<<"error in reding file"<<std::endl;
    }

    std::getline(infile, data);
	if(data.compare(0, 2, "P6")!=0){
		std::cout<<"File type "<<data<<"is not expected data type"<<std::endl;	
	}

	do{
		std::getline(infile, data);
	}while(data.compare(0, 1, "#")==0);
	
	//reading dimension of size
	size_t i1 = data.find(" ");
	if(i1!=std::string::npos){
		X = std::stoi(data.substr(0,i1));
		Y = std::stoi(data.substr(i1));
	}

	//assign character pointer of image size
	unsigned char* fdata = (unsigned char*) malloc(X * Y * 3 * sizeof(unsigned char));
	//escape next line
	std::getline(infile, data);
	char m; 
	size_t i =0;
	for(int i = 0; i < X * Y * 3; i++){
		  infile.get(m);
		  fdata[i] = (unsigned char) m;
    }
	infile.close();
	return fdata;

}
void writeImg(std::string outfilename, unsigned char* filedata, size_t width, size_t height){
	char strtemp[33];
	std::ofstream output(outfilename, std::ios::binary|std::ios::out);
	if(!output){
		std::cout << " unable to open the output file "<< outfilename <<std::endl;
	}
	else{
		output << "P6"<<std::endl;
		output << itoa(width, strtemp, 10);
		output << " ";
		output << itoa(height, strtemp, 10);
		output << std::endl;
		output << itoa(255, strtemp, 10) << std::endl;
		output.write( (char *)filedata, width * height * 3);
		output.close();

	};
}

void xconvole(unsigned char* outdata, unsigned char* indata, float* gxkernel, size_t xout, size_t yout){
//this function performs convolution in X dimension on CPU
	//ix and iy are pixel indices of output data in X and Y dimension 
	for(size_t iy = 0; iy< Y; iy++ ){
		for(size_t ix = 0; ix< xout; ix++){
			//intialie temporary varialbles to hold a value of current pixel red, gree, blue channels to zero
			 float temp0 = 0;
			 float temp1 = 0;
			 float temp2 = 0;
			 			
			 //perform convolution on a pixels red, green and blue channels
			 size_t i = iy * xout + ix;										//this gives row major index of output image
			 //As shorter image is produced it is possible to find out first pixel position of current windowed data in input image the data 
			 size_t in_px = iy* X +ix;										
			 
			 //perform kernel image and input image multiplication for convolution result of current pixel in output image
			 for(size_t n = 0; n< k; n++){
				// if (in_px + n >= 0 && in_px + n < k) {                   //to check boundries when output image is same as original image 
					size_t redind = (in_px+n)* 3;							//pixel index of input image
					float gkernel = gxkernel[n];
					temp0 = temp0 + indata[redind] * gkernel;			//red channel 
					temp1 = temp1 + indata[redind + 1] * gkernel;		//blue channel
					temp2 = temp2 + indata[redind + 2] * gkernel;       //green channel
				 //}
			 }
			 
		    outdata[i * 3 + 0] = (unsigned char)temp0;						//write convolution result of red channel of current pixel in output image
			outdata[i * 3 + 1] = (unsigned char)temp1;						//write convolution result of green channel of current pixel in output image
			outdata[i * 3 + 2] = (unsigned char) temp2;						//write convolution result of blue channel of current pixel in output image
		}
	}
}

void yconvole(unsigned char* outdata, unsigned char* indata, float* gxkernel, size_t xout, size_t yout){
//this performs convolution in Y dimesion as kernel is same in both X and Y dimension, So same xkernel is used here as well

	for(size_t ix = 0; ix< xout; ix++ ){
		for(size_t iy = 0; iy< yout; iy++){
			 // intialize temprory pixel result to zero
			 float temp0 = 0;
			 float temp1 = 0;
			 float temp2 = 0;
			 			
			 //perform convolution for a pixel's red, green and blue channel in each loop

			 size_t i = iy * xout + ix;													//row major index in 1D output image data
			 //As shorter image is produced it is possible to find out first pixel position of current windowed data in input image the data 
			 size_t in_px = iy * xout + ix;			
			 //return if threads are going out of image size
			 if(ix >= xout || iy >= yout) return;

			 //perform kernel and input image multiplication
			 for(size_t n = 0; n< k; n++){
					size_t redind = (in_px+n*xout)* 3;									//imput image pixel index							
					float gkernel = gxkernel[n];
					temp0 = temp0 + indata[redind] * gkernel;						//red channel
					temp1 = temp1 + indata[redind + 1] * gkernel;					//blue channel
					temp2 = temp2 + indata[redind + 2] * gkernel;					//green channel
			 }
			
			//write result for a pixel to output image
		    outdata[i * 3 + 0] = (unsigned char)temp0;									//write convolution result of a red channel pixel
			outdata[i * 3 + 1] = (unsigned char)temp1;									//write convolution result of a blue channel pixel 
			outdata[i * 3 + 2] = (unsigned char) temp2;									//write convolution result of a green channel pixel
		}
	}
}

__global__ void kernelxconvole(unsigned char* outdata, unsigned char* indata, float* gxkernel, size_t xout, size_t Y, size_t k, size_t X){
	//This function performs convolution in X dimension on GPU
	//ix and iy are pixel indices of 2d output image
	size_t ix, iy;
	ix = blockIdx.x * blockDim.x + threadIdx.x;
	iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix >= xout || iy >= Y) return;
			//intialie temporary varialbles to hold a value of current pixel red, gree, blue channels to zero
			 float temp0 = 0;
			 float temp1 = 0;
			 float temp2 = 0;
			 			
			 //perform convolution on a pixels red, green and blue channels
			 size_t i = iy * xout + ix;										//this gives row major index of output image
			 //As shorter image is produced it is possible to find out first pixel position of current windowed data in input image the data 
			 size_t in_px = iy* X +ix;										
			 
			 //perform kernel image and input image multiplication for convolution result of current pixel in output image
			 for(size_t n = 0; n< k; n++){
				// if (in_px + n >= 0 && in_px + n < k) {                   //to check boundries when output image is same as original image 
					size_t redind = (in_px+n)* 3;							//pixel index of input image
					float gkernel = gxkernel[n];
					temp0 = temp0 + indata[redind] * gkernel;			//red channel 
					temp1 = temp1 + indata[redind + 1] * gkernel;		//blue channel
					temp2 = temp2 + indata[redind + 2] * gkernel;       //green channel
				 //}
			 }
			 
		    outdata[i * 3 + 0] = (unsigned char)temp0;						//write convolution result of red channel of current pixel in output image
			outdata[i * 3 + 1] = (unsigned char)temp1;						//write convolution result of green channel of current pixel in output image
			outdata[i * 3 + 2] = (unsigned char) temp2;						//write convolution result of blue channel of current pixel in output image
		
}

__global__ void kernelyconvole(unsigned char* outdata, unsigned char* indata, float* gxkernel, size_t xout, size_t yout, size_t k){
	//This function performs convolution in y dimension on GPU
	//ix and iy are pixel indices of 2d output image
	size_t ix, iy;
	ix = blockIdx.x * blockDim.x + threadIdx.x;
	iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix >= xout || iy >= yout) return;
			 // intialize temprory result to zero
			 float temp0 = 0;
			 float temp1 = 0;
			 float temp2 = 0;
			 			
			 //perform convolution on a pixel's of red, green and blue channel in each loop

			 size_t i = iy * xout + ix;													//row major index in 1D output image data
			 //As shorter image is produced it is possible to find out first pixel position of current windowed data in input image the data 
			 size_t in_px = iy * xout + ix;			
			 //return if threads are going out of image size
			
			
			 //perform kernel and input immage multiplication
			 for(size_t n = 0; n< k; n++){
					size_t redind = (in_px+n*xout)* 3;									// input image pixel index
					float gkernel = gxkernel[n];
					temp0 = temp0 + indata[redind] * gkernel;						// red channel
					temp1 = temp1 + indata[redind + 1] * gkernel;					// green
					temp2 = temp2 + indata[redind + 2] * gkernel;					// blue
			 }
			
			//write result for a pixel to output image
		    outdata[i * 3 + 0] = (unsigned char)temp0;	//write convolution result of a red channel pixel
			outdata[i * 3 + 1] = (unsigned char)temp1;	//write convolution result of a blue channel pixel 
			outdata[i * 3 + 2] = (unsigned char) temp2;	//write convolution result of a green channel pixel
}



int main(int argc, char* argv[]){
	
	std::cout<<"Welcome to 2d separable convolution!!!"<<std::endl; 
	std::cout<<std::endl;
	if(argc!=4){
		std::cout<<"Error in command line arguments"<<std::endl; 
		std::cout<<"format : convolve infile outfile std "<<std::endl;
		return 1;	
	}

	std::string filename = argv[1];													//input image filename from command line
	std::string outfilename = argv[2];												//output image filename from command line
	
	std::cout<<"input image is: "<<filename<<std::endl;
	std::cout<<"output image is: "<<outfilename<<std::endl;
	std::cout<<std::endl;

	//reading ppm image 
	unsigned char* fdata = readImg(filename);
	
	//===========create kernel=============================
	int sigma = atoi(argv[3]);														//read std deviation from command line
	k = 6*sigma;																	//acquire kernel size from sigma value
	if(k%2==0)k++;
	float* gxkernel = (float*)malloc(k * sizeof(float));
	createKernel(gxkernel, sigma, k);												//calling function to create kernel

		
	//===========separable convolution on CPU  ======================
	//calculate output image size
	size_t xout = X - k +1 ;
	size_t yout = Y - k + 1;
	
	//allocate memory for resulting images in x dimension and y dimension
	unsigned char* xdata = (unsigned char*)malloc(xout*Y*3*sizeof(unsigned char));
	unsigned char* ofdata = (unsigned char*)malloc(xout*Y*3*sizeof(unsigned char));
	
	//---------------------------------convole with x dimesion kernel
	unsigned int start = time(NULL);												//start timer
	xconvole(xdata, fdata, gxkernel, xout, yout);
		
	//---------------------------------convole with y dimesion kernel
	yconvole(ofdata, xdata, gxkernel, xout, yout);
	unsigned int end = time(NULL);													//stop timer
	std::cout<<"convolution time on cpu is "<<end-start<<"s"<<std::endl;			//total time for convolutio on cpu

	//================separable convolution on GPU=======================
	
	//check gpu properties
	cudaDeviceProp props;										//declare a CUDA properties structure
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));			//get the properties of the first CUDA device
	
	//allocate memory for input image and resulting image in x dimension convolution 
	unsigned char* gpu_fdata;
	unsigned char* gpu_xdata;
	float* gpu_Gxkernel;

	HANDLE_ERROR(cudaMalloc(&gpu_fdata, X*Y*3*sizeof(unsigned char)));											//allocate device memory for input imge
	HANDLE_ERROR(cudaMemcpy(gpu_fdata, fdata, X*Y*3*sizeof(unsigned char), cudaMemcpyHostToDevice));			//copy from host to device
	HANDLE_ERROR(cudaMalloc(&gpu_Gxkernel, k*sizeof(float)));													//allocate device memory for kernel
	HANDLE_ERROR(cudaMemcpy(gpu_Gxkernel, gxkernel, k*sizeof(float), cudaMemcpyHostToDevice));					//copy from host to device

	HANDLE_ERROR(cudaMalloc(&gpu_xdata, xout*Y*3*sizeof(unsigned char)));										//allocate device memory for output image
		
	//---------------------convolution in  X dimension
	//compute block and grid dimension 
	unsigned int threads = (unsigned int)sqrt(props.maxThreadsPerBlock);
	dim3 blockdim(threads, threads);
	dim3 griddim(xout/threads +1,Y/threads +1 );
	
	unsigned int startg = time(NULL);
	kernelxconvole <<<griddim, blockdim>>> (gpu_xdata, gpu_fdata, gpu_Gxkernel, xout, Y, k, X);					//launch the kernel for x dimension convolution
	cudaFree(gpu_fdata);																						//cuda free for original input image
	
	//---------------------convolution in y dimension
	unsigned char* gpu_ofdata;																					//gpu pointer for final convolved output
	HANDLE_ERROR(cudaMalloc(&gpu_ofdata, xout*yout*3*sizeof(unsigned char)));									//allocate memory for final output image

	//compute block dimension and grid dimension 
	dim3 blockdim_y(threads, threads);
	dim3 griddim_y(xout/threads + 1, yout/threads + 1 );

	kernelyconvole<<<griddim_y, blockdim_y>>>(gpu_ofdata, gpu_xdata, gpu_Gxkernel, xout, yout, k);				//launch the kernel for y dimension convolution
	 
	HANDLE_ERROR(cudaMemcpy(ofdata, gpu_ofdata, xout*yout*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));	//copy results from kernel to host
	
	//free all gpu variables
	cudaFree(gpu_xdata);
	cudaFree(gpu_ofdata);
	cudaFree(gpu_Gxkernel);
	
	unsigned int endg = time(NULL);																				//stop timer
	std::cout<<"convolution time on gpu is "<<endg-startg<<"s"<<std::endl;										//compute total time for convolution
	cudaDeviceReset();     //reset gpu device
	
	writeImg(outfilename, ofdata, xout, yout);																	//writing gpu blur data to output file 
	std::cout<<"Blurring is done"<<std::endl;
	
	//free all cpu pointers
	if(fdata!=NULL)
		free(fdata);                          
	if(xdata!=NULL)
		free(xdata);
	if(ofdata!=NULL)
		free(ofdata); 
	if(gxkernel!=NULL)
		free(gxkernel);
		    
}
