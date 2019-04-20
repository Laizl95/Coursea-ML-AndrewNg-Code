#include<bits/stdc++.h>
#define rep(i,x,y) for(int i=x;i<=y;++i)
#define per(i,x,y) for(int i=y;i>=x;--i)
#define ms(x,y) memset(x,y,sizeof(x))
typedef long long LL;
using namespace std;
const int N=85;
int ans[N],n,m;
int dfs(int &pos,int &num){
    //printf("%d\n",num);
    if(num==n) return 1;
    rep(i,0,m-1){
        ans[pos]=i;int flag=0;
        int r=(pos+1)/2;
       rep(j,1,r){
           int l=pos;flag=1;
         for(int k=0;k<j;++k){
            if(ans[l]!=ans[l-j]){flag=0;break;}
            l-=1;}
         if(flag) break;
       }
       if(flag==0){
            num+=1;pos+=1;
        if(dfs(pos,num)) return 1;
            pos-=1;
       }
    }
    return 0;
}
int main(){
    while(scanf("%d %d",&n,&m)!=EOF){
        if(!n && !m) break;
        int pos=0,num=0;
        if(dfs(pos,num)){
            rep(i,0,pos-1){
             printf("%c",ans[i]+'A');
             if((i+1)%64==0) puts("");
             else if(i!=pos-1 && (i+1)%4==0) printf(" ");
            }
        }
        if(pos%64!=0) puts("");
        printf("%d\n",pos);
    }
return 0;
}
