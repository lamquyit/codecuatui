#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

int main (){
	int t ;
	cin >> t ;
	while (t--){
		string s;
		cin >> s ;
		int tmp =s.find("084");
		cout << s.erase(tmp,3)<<endl;
	}
}
