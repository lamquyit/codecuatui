#include <bits/stdc++.h>

using namespace std ;

long gcd(long a, long b){
	if (b == 0) return a;
	else return gcd(b,a%b);
}



long lcm(long a , long b){
	return a / gcd (a,b)*b;
}

int main (){
	int m;
	cin >> m ; 
	while (m--){
	long long x , y ;
	cin >> x >> y ;
	cout << lcm(x , y) << " " << gcd( x , y)  << endl;
	}
}
