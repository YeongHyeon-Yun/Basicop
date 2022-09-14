#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;
void PointOp();                   //픽셀의 좌표를 표현하는 클래스
void SizeOp();                    //사각형 영역의 크기
void RectOp();                    //사각형의 위치와 크기 정보
void RotateRectOp();              //회전된 사각형
void RangeOp();                   //범위 또는 구간을 표현
void StringOp();                  //문자열
int main()
{
	//PointOp();
	//SizeOp();
	//RectOp();
	//RotateRectOp();
	RangeOp();
	//StringOp();
}
void PointOp()
{
	Point pt1;				// pt1 = (0, 0)
	pt1.x = 5; pt1.y = 10;	// pt1 = (5, 10)
	Point pt2(10, 30);		// pt2 = (10, 30)
	Point pt3 = pt1 + pt2;	// pt3 = [15, 40]
	Point pt4 = pt1 * 2;	// pt4 = [10, 20]
	int d1 = pt1.dot(pt2);	// d1 = 350
	bool b1 = (pt1 == pt2);	// b1 = false
	cout << "pt1: " << pt1 << endl;
	cout << "pt2: " << pt2 << endl;
}
void SizeOp()
{
	Size sz1, sz2(10, 20);			// sz1 = [0 x 0], sz2 = [10 x 20]
	sz1.width = 5; sz1.height = 10;	// sz1 = [5 x 10]
	Size sz3 = sz1 + sz2;	// sz3 = [15 x 30]
	Size sz4 = sz1 * 2;		// sz4 = [10 x 20]
	int area1 = sz4.area();	// area1 = 200
	cout << "sz3: " << sz3 << endl;
	cout << "sz4: " << sz4 << endl;
}
void RectOp()
{
	Rect rc1;                          // rc1 = [0 x 0 from (0, 0)]
	Rect rc2(10, 10, 60, 40);          // rc2 = [60 x 40 from (10, 10)]
	Rect rc3 = rc1 + Size(50, 40);     // rc3 = [50 x 40 from (0, 0)]
	Rect rc4 = rc2 + Point(10, 10);    // rc4 = [60 x 40 from(20, 20)]
	Rect rc5 = rc3 & rc4;              // rc5 = [30 x 20 from (20, 20)]
	Rect rc6 = rc3 | rc4;              // rc6 = [80 x 60 from (0, 0)]
	cout << "rc5: " << rc5 << endl;
	cout << "rc6: " << rc6 << endl;
}
void RotateRectOp()
{
	/* RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);
	Point2f pts[4];
	rr1.points(pts);
	Rect br = rr1.boundingRect();
	*/
	Mat test_image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 50), 45.f);
	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(test_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
	Rect brect = rRect.boundingRect();
	rectangle(test_image, brect, Scalar(255, 0, 0), 2);
	imshow("rectangle", test_image);
	waitKey(0);
}
void RangeOp() {
	Mat img;
	Mat rect_img;
	img = imread("lenna.bmp");
	int y1 = 220;
	int y2 = 320;
	int x1 = 200;
	int x2 = 370;
	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	rect_img = img(Range(y1, y2), Range(x1, x2));
	namedWindow("원본이미지", WINDOW_NORMAL);
	namedWindow("결과이미지", WINDOW_AUTOSIZE);
	imshow("원본이미지", img);
	imshow("결과이미지", rect_img);
	waitKey();
	destroyAllWindows();
}
void StringOp()
{
	String str1 = "Hello";
	String str2 = "world";
	String str3 = str1 + " " + str2;	// str3 = "Hello world"
	bool ret = (str2 == "WORLD");
	Mat imgs[3];
	for (int i = 0; i < 3; i++) {
		String filename = format("data%02d.bmp", i + 1);
		cout << filename << endl;
		imgs[i] = imread(filename);
	}
}
#endif


#if 1
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

void MatOp1();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);	// 로깅 메세지들 띄우지 않기
	MatOp1();
}
void MatOp1()
{
	Mat img1;  //empty matrix
	// 640 x 480
	Mat img2(480, 640, CV_8UC1);   // Unsigned Channel
	Mat img3(480, 640, CV_8UC3);
	Mat img4(Size(640, 480), CV_8UC3);
	Mat img5(480, 640, CV_8UC1, Scalar(255));	// 스칼라 값 0: 검은색, 128: 회색, 255: 흰색


	/*
	namedWindow("image1");
	imshow("image1", img5);
	waitKey();
	*/

	Mat img6(480, 640, CV_8UC3, Scalar(150, 100, 100));

	/*
	namedWindow("image2");
	imshow("image2", img6);
	waitKey();
	*/

	Mat mat1 = Mat::zeros(3, 3, CV_8SC1);
	Mat mat2 = Mat::ones(3, 3, CV_32FC1);
	Mat mat3(3, 3, CV_32FC1, Scalar(6.25));
	Mat mat4 = Mat::eye(3, 3, CV_8SC1);

	float data1[] = { 1, 2, 3, 4, 5, 6 };
	Mat mat5(3, 2, CV_32FC1, data1);

	data1[0] = 100;
	data1[3] = 300;
	
	Mat_<float> mat5_(2, 3);
	mat5_ << 1, 2, 3, 4, 5, 6;
	Mat mat6 = mat5_;

	Mat mat7 = (Mat_<float>(2, 3) << 1, 2, 3, 4, 5, 6);

	Mat mat8 = Mat_<float>({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });



}
#endif