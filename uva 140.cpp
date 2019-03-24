#include<bits/stdc++.h>
#define rep(i,x,y) for(int i=x;i<=y;++i)
#define per(i,x,y) for(int i=y;i>=x;--i)
#define ms(x,y) memset(x,y,sizeof(x))
typedef long long LL;
using namespace std;
const int N=205;
char s[N];
int vis[28],ans[28],cnt,finals[28];
vector<int> child[28];
vector<int> fa;
void solve(){
  int len=strlen(s),u=-1;
  for(int i=0;i<len;++i){
    if(u==-1){
        u=s[i]-'A';
        vis[u]=1;
        continue;
    }
    if(s[i]==':') continue;
    if(s[i]==';'){u=-1;continue;}
    child[u].push_back(s[i]-'A');
    child[s[i]-'A'].push_back(u);
    vis[s[i]-'A']=1;
  }
}
int find_pos(int x){
  rep(i,0,cnt) if(ans[i]==x) return i;
}
int get(int x,int pos){
 int t=child[x].size()-1,ans=0;
 rep(i,0,t)
  ans=max(ans,abs(pos-find_pos(child[x][i])));
 return ans;
}
int dis(int x,int pos,int k){
  int y=child[x].size()-1,visit[9];
  ms(visit,0);
  rep(i,0,y) visit[child[x][i]]=1;
  rep(i,0,pos-1) if(visit[ans[i]]&& abs(i-pos)>k) return 0;
  return 1;
}
int dfs(int u,int &min_bands,int pos){
    //rep(i,0,pos-1) {printf("%c ",ans[i]+'A');}puts("");
    if(pos==cnt+1){
            int min_sub=-1;
        rep(i,0,cnt){
           min_sub=max(min_sub,get(ans[i],i));
        }
        if(min_bands>min_sub){
            rep(i,0,cnt) finals[i]=ans[i];
            min_bands=min_sub;
        }
        return 1;
    }
    if(u!=-1){if(child[u].size()>min_bands) return 0;}
    rep(i,0,cnt){
        if(vis[fa[i]] || !dis(fa[i],pos,min_bands)) continue;
        ans[pos]=fa[i];
        vis[fa[i]]=1;
        dfs(fa[i],min_bands,pos+1);
        vis[fa[i]]=0;
    }
    return 1;
}
int main(){
   // freopen("e://input.txt","r",stdin);
   while(scanf("%s",s)!=EOF && s[0]!='#'){
        ms(vis,0);
        solve();
        rep(i,0,25)
        if(vis[i]){
            fa.push_back(i);
            int x=child[i].size();
            sort(child[i].begin(),child[i].end());
        }
        ms(vis,0);
        cnt=fa.size()-1;
        int min_bands=10;
        dfs(-1,min_bands,0);
        rep(i,0,cnt) printf("%c ",finals[i]+'A');
        printf("-> %d\n",min_bands);
        rep(i,0,cnt) child[fa[i]].clear();
        fa.clear();
   }
return 0;
}
