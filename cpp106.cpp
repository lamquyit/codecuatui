#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;

using ll = long long ;

bool thuannghich(ll n ){
	ll reverse = 0 , tmp = n ;
	while (n>0){
		reverse = reverse * 10 + n % 10;
		n/=10;	
	}	
	return reverse == tmp;
}

int main (){
	int n ;
	cin >> n ;
	while (n --){
		ll m ;
		cin >> m ;
		if(thuannghich(m)){
			cout << "YES"<< endl;
			}else cout << "NO" <<endl;
		}
}
