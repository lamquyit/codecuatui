#include<bits/stdc++.h> 
using namespace std; int ngto(int a) { if(a<2) return 0; for(int i=2;i<=sqrt(a);i++){ if(a%i==0) return 0; } return 1; } int main(){ int a; cin >> a ; while(a--){ int n; cin>>n; int check = 0; for(int i=2;i<=n/2;i++){ if(ngto(i)&&ngto(n-i)){ cout << i << " " << n-i<<endl; check =1; break; } } if(check == 0 ) { cout << "-1\n"; } } }
