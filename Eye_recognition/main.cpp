#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

String face_cascade_name = "E:/prog/other/opencv/data/lbpcascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = "E:/prog/other/opencv/data/haarcascades/haarcascade_eye.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

RNG rng(12345);


int main( int argc, const char** argv )
{
  CvCapture* capture;
  Mat frame;


  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  
  capture = cvCaptureFromCAM( -1 );
  if( capture )
  {
    while( true )
    {
      frame = cvQueryFrame( capture );
  
      
      if( !frame.empty() )
       { detectAndDisplay( frame ); }
      else
       { printf(" --(!) No captured frame -- Break!"); break; }
      
      int c = waitKey(10);
      if( (char)c == 'c' ) { break; } 

    }
  }
  return 0;
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
      Mat faceROI=frame_gray( faces[i] ); ;
/*	  cvtColor(frame_gray( faces[i]),faceROI,CV_BGR2GRAY);
	  equalizeHist(faceROI ,faceROI );*/
      std::vector<Rect> eyes;
	  Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
      ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );

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
