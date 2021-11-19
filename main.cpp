#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <random>
#include <vector>
#include <string>
//#include <sstream>
//#include <iterator>

#include <stim/image/image.h>

#include <opencv2/opencv.hpp>
#include <ml.h>
#include <cv.h>
#include <highgui.h>

//using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
	
	std::cout<<"number of arguments"<<argc<<std::endl;
	if(argc!=4){
		std::cout<<"Error in command line arguments"<<std::endl; 
		std::cout<<"format : convolve infile outfile std "<<std::endl;
		return 1;	
	}

	//user parameters from command line
	std::string filename = argv[1];
	std::string outfilename = argv[2];
	//std::cout<<"outfilename: "<<outfilename<<std::endl;
	int sigma = atoi(argv[3]);
	int k = 6*sigma;                    //acquire kernel size from sigma value
	if(k%2==0)k++;
	float* kernel = (float*)malloc(k * sizeof(float));

	//reading file
	stim::image<unsigned char> I(filename);
	/*cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0);*/
	stim::image<unsigned char> red = I.channel(0);
	std::vector< stim::image<unsigned char> > vec(I.channels());
	for(unsigned int cl = 0 ; cl < I.channels() ; cl++)
		vec[cl] = I.channel(cl);							// load each channel of the target image as a vecotor element

	unsigned int a0 = (unsigned int)vec[0].data()[0];
	unsigned int a1 = (unsigned int)vec[1].data()[0];
	unsigned int a2 = (unsigned int)vec[2].data()[0];
	unsigned char *ptr = I.data();
	for(int k = 0; k<20; k++)
	cout<<unsigned int(ptr[k])<<endl;

     
}
