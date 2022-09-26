#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

float khoangcach(float a , float b  , float c ,float d ){
	float kc;
	kc =  sqrt(pow(a-c,2) + pow(b-d,2));
	return kc;
}

int main (){
	int m ;
	cin >> m ;
	while (m --){
		float  a , b , c , d;
		cin >> a >> b >> c >> d ;
		cout<< fixed << setprecision(4) << khoangcach(a, b,c ,d) << endl;
	}
}
