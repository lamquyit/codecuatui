#include <bits/stdc++.h>

using namespace std ;

int main (){
	int t ;
	double khoangcach;
	cin >> t ;
	while (t --){
		double a,b,c,d;
		cin >> a >> b >> c >> d;
		khoangcach = sqrt((b-d)*(b-d)+(a-c)*(a-c));
		cout << fixed << setprecision (4)<< khoangcach << endl ;
	}
	return 0 ;
}
