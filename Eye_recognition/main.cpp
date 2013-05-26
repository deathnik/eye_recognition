#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

// to access protected fields of CascadeClassifier
class CascadeClassifierChild : public CascadeClassifier{
public:
	int UseRunAt(Point pt, double& weight){
		return runAt(featureEvaluator, pt,weight);
	}
};
//first rect- face, vector contains two eyes
typedef std::pair<Rect, std::vector<Rect>> Face;


int doGood(String face_cascade_name,String eyes_cascade_name,double checkRate,int updateDelay);
void fullReload(Mat frame,std::vector<Face> * result);
void display(std::vector<Face> * faces,Mat frame);
bool check(double checkRate,std::vector<Face> * faces,Mat frame);
void detectAndDisplay( Mat frame );


CascadeClassifierChild face_cascade;
CascadeClassifierChild eyes_cascade;
RNG rng(12345);


int main( int argc, const char** argv )
{
	String face_cascade_name = "../cascades/lbpcascade_frontalface.xml";
	String eyes_cascade_name = "../cascades/haarcascade_eye.xml";
	double checkRate= 0.05;
	int updateDelay =25;
	return doGood(face_cascade_name,eyes_cascade_name,checkRate,updateDelay);
}

int doGood(String face_cascade_name,String eyes_cascade_name,double checkRate,int updateDelay){
  CvCapture* capture;
  Mat frame;

  if( !face_cascade.load( face_cascade_name ) ){
	  printf("--(!)Error loading face cascade\n");
	  return -1;
  };

  if( !eyes_cascade.load( eyes_cascade_name ) ){
	  printf("--(!)Error loading eyes cascade\n");
	  return -1;
  };

  
  capture = cvCaptureFromCAM( -1 );
  std::vector<Face>* faces = new vector<Face>();
  if( capture )
  {
    while( true )
    {
      frame = cvQueryFrame( capture );
  
      
      if( !frame.empty() )
       {   
		   if(faces->size() == 0 ){
			   fullReload(frame,faces);
		   }
		   else{
				if(!check(checkRate,faces,frame)){
					fullReload(frame,faces);
				}
		   }
		   display(faces, frame ); 
	  }
      else
       {
		   printf(" --(!) No captured frame -- Break!");
		   break;
	  }
      
      int c = waitKey(updateDelay);
      if( (char)c == 'c' ) { 
		  break;
	  } 
    }
  }
  return 0;
}

void fullReload(Mat frame,std::vector<Face> * result){
   //get possible face areas
   result->clear();
   std::vector<Rect> faces;
   Mat frame_gray=frame;
   cvtColor( frame, frame_gray, CV_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0 , Size(80, 80) );
   //find eyes on each face
   if(faces.size() == 0){
	   return ;
   }
   for( int i = 0; i < faces.size(); i++ ){
	   Mat faceROI=frame_gray( faces[i] ); 
       std::vector<Rect> eyes;
	   eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 , Size(30, 30) );
	   if(eyes.size() == 2){
		   result->push_back(Face(faces[i],eyes));
	   }
   }
}

void display(std::vector<Face> * faces,Mat frame ){
	//Draw each face
	for(int i =0; i < faces->size(); i++ ){
		Rect * faceRect = &(*faces)[i].first; 
		Point center( faceRect->x + faceRect->width*0.5, faceRect->y + faceRect->height*0.5 );
        ellipse( frame, center, Size( faceRect->width*0.5, faceRect->height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
		//Draw the eyes on face
		std::vector<Rect> * eyes =&(*faces)[i].second;
		for( int j = 0; j < eyes->size(); j++ ){ 
			Rect * eyeRect = &(*eyes)[j]; 
            Point center( faceRect->x +eyeRect->x + eyeRect->width*0.5, faceRect->y + eyeRect->y + eyeRect->height*0.5 ); 
            int radius = cvRound( (eyeRect->width + eyeRect->height)*0.25 );
            circle( frame, center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
          }
	}
	imshow( "", frame );
}

bool check(double checkRate,std::vector<Face> * faces,Mat frame){
	if(faces->size() == 0) return false;
	face_cascade.setImage(frame);
	for(int i = 0; i< faces->size(); i++){
		Rect * faceRect = &(*faces)[i].first; 
		if(!face_cascade.UseRunAt(faceRect->tl(),checkRate))return false;
	}
	return true;
}


void detectAndDisplay( Mat frame )
{
   std::vector<Rect> faces;
   Mat frame_gray=frame;

   cvtColor( frame, frame_gray, CV_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );

   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0 , Size(80, 80) );

   for( int i = 0; i < faces.size(); i++ )
    {
	  
      Mat faceROI=frame_gray( faces[i] ); 
      std::vector<Rect> eyes;
	  Rect * faceRect = &faces[i]; 
		Point center( faceRect->x + faceRect->width*0.5, faceRect->y + faceRect->height*0.5 );
        ellipse( frame, center, Size( faceRect->width*0.5, faceRect->height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
		

      //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 , Size(30, 30) );
	  
         for( int j = 0; j < eyes.size(); j++ )
          { //-- Draw the eyes
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 ); 
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
          }
       

    } 
   //-- Show what you got
   imshow( "", frame );
}
