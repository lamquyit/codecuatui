#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;

double tinhdiem(char a[], char b[]){
	float diem = 0 ; 
	for (int i = 0 ; i < 15 ;i ++){
	if ( a [i] == b [i]){
		diem += 2.0/3;
		}
	}
	return diem ;
}

int main (){
	char a [15] = {'A','B','B','A','D','C','C','A','B','D','C','C','A','B','D'};
	char b [15] = {'A','C','C','A','B','C','D','D','B','B','C','D','D','B','B'};
	int n ;
	cin >> n ;
	while (n --){
		int x ;
		cin >> x ;
		char da[15];
		for (int i = 0 ; i < 15 ; i ++ ){
			cin >> da[i];
		}
		if (x == 101){
			cout << fixed << setprecision(2) << tinhdiem(da, a)<< endl;
		}else if(x == 102 ) {
			cout << fixed << setprecision(2) << tinhdiem(da, b)<< endl;
		}
	}
}
