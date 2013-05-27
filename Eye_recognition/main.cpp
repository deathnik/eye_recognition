#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;
RNG rng(12345);
// to access protected fields of CascadeClassifier
class CascadeClassifierChild : public CascadeClassifier{
public:
	int UseRunAt(Point pt, double& weight){
		if(featureEvaluator.empty() )return 0;
		return runAt(featureEvaluator, pt,weight);
	}
};
//first rect- face, vector contains two eyes
typedef std::pair<Rect, std::vector<Rect>> Face;

class eye_finder{
protected:
	CascadeClassifierChild face_cascade;
	CascadeClassifierChild eyes_cascade;
	double * average;
	double checkRate;
	double updateDelay;

	double learningRate;

public:
	eye_finder(String face_cascade_name,String eyes_cascade_name,double _checkRate,int _updateDelay){
		learningRate = 0.1;
		checkRate= _checkRate;
		updateDelay=_updateDelay;
		average = new double [8];
		for(int i =0 ; i < 8; i ++){
			average[i]=0.0;
		}
		

		if( !face_cascade.load( face_cascade_name ) ){
			printf("--(!)Error loading face cascade\n");
			return ;
		};

		if( !eyes_cascade.load( eyes_cascade_name ) ){
			printf("--(!)Error loading eyes cascade\n");
			return ;
		};
	}

	int doGood(){
		CvCapture* capture;
		Mat frame;

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
						if(!check(faces,frame)){
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

private:

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
		   std::vector<Rect> eyes = vector<Rect>();
		   tryToFindEyes(&eyes,&faces[i]);
		   if(eyes.size()!=2)eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 , Size(30, 30) );
	   
		   if(eyes.size() == 2){
			   for(int k=0; k< eyes.size(); k++){
				  average[k*4] =  average[k*4]<=0.0000001 ? eyes[k].tl().x/(double)faces[i].width : eyes[k].tl().x*learningRate/(double)faces[i].width + average[k*4]*(1.0 - learningRate) ;
				  average[k*4+1] = average[k*4+1]<=0.000001 ? eyes[k].tl().y/(double)faces[i].height : eyes[k].tl().y*learningRate/(double)faces[i].height + average[k*4+1]*(1.0 - learningRate) ;
				  average[k*4+2] =  average[k*4+2]<=0.0000001 ? eyes[k].width : eyes[k].width*learningRate + average[k*4+2]*(1.0 - learningRate) ;
				  average[k*4+3] = average[k*4+3]<=0.000001 ? eyes[k].height: eyes[k].height*learningRate + average[k*4+3]*(1.0 - learningRate) ;
			   }
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
	


	bool check(std::vector<Face> * faces,Mat frame){
		if(faces->size() == 0) return false;
		face_cascade.setImage(frame);
		for(int i = 0; i< faces->size(); i++){
			Rect * faceRect = &(*faces)[i].first; 
			if(!face_cascade.UseRunAt(faceRect->tl(),this->checkRate))return false;
		}
		return true;
	}


	void tryToFindEyes(std::vector<Rect> *eyes,Rect * face){
		Point tmp =face->tl()+Point(average[0]*face->width,average[1]*face->height);
		bool leftEye= eyes_cascade.UseRunAt(face->tl()+Point(average[0]*face->width,average[1]*face->height),this->checkRate);
		bool rightEye = eyes_cascade.UseRunAt(face->tl()+Point(average[4]*face->width,average[5]),this->checkRate);
		if(leftEye && rightEye){
			eyes->push_back(Rect(average[0],average[1],average[2],average[3]));
			eyes->push_back(Rect(average[4],average[5],average[6],average[7]));
		}
	}

};

int main( int argc, const char** argv )
{
	String face_cascade_name = "../cascades/lbpcascade_frontalface.xml";
	String eyes_cascade_name = "../cascades/haarcascade_eye(2).xml";
	double checkRate= 0.5;
	int updateDelay =25;
	eye_finder finder = eye_finder(face_cascade_name,eyes_cascade_name,checkRate,updateDelay);
	return finder.doGood();
}