#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <limits>
#include <string>
#include <utility>
#include <sys/stat.h>  
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using MapType = std::map<std::pair<int, int>, std::vector<std::vector<uint64_t>>>;
using PairType = std::pair<const std::pair<int, int>, std::vector<int>>;

int BIG_SIZE=1000000000;
int small_size=500;
//using namespace chrono;
using namespace std;
struct one_matrix
{
    map<pair<int,int>,vector<vector<uint64_t>>>mat;
    int rows;
    int cols;
    int blocks;
};
#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t err = (call);                                                       \
    if (err != cudaSuccess) {                                                       \
      fprintf(stderr, "CUDA error at %s:%d: `%s` failed: %s\n",                     \
              __FILE__, __LINE__, #call, cudaGetErrorString(err));                 \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  } while (0)


__global__ void matrix_multiplyKernel(uint64_t* large_arr_gpu,int *key_to_elem_gpu,int *key_to_elem_prefix_gpu,int k_given,uint64_t* out_arr_gpu){
    int block_ind =blockIdx.x;
    //int tx =threadIdx.x+block_ind*k_given;
    int ty=threadIdx.y;
    const uint64_t MAX = 18446744073709551615ULL;
    int num_matrices =key_to_elem_gpu[block_ind];
    int offset =key_to_elem_prefix_gpu[block_ind]*2*k_given*k_given;
    //int elem1=large_arr[offset+2*i*k_given*k_given+ty*k_given +j];
    //int elem2=large_arr[offset+2*i*k_given*k_given+k_given*k_given+j*k_given+threadidx.x]
    uint64_t total_sum=0;
    for(int i=0;i<num_matrices;i++){
        
        for(int j=0;j<k_given;j++){
            uint64_t elem1=large_arr_gpu[offset + 2*i*k_given*k_given + ty*k_given + j];
            uint64_t elem2=large_arr_gpu[offset + 2*i*k_given*k_given + k_given*k_given + j*k_given+threadIdx.x];
            uint64_t total_sum1=(elem1*elem2)%MAX;
           // total_sum1=total_sum1%MAX;
            total_sum=(total_sum+total_sum1)%MAX;
        }
    } 
    __syncthreads();
    out_arr_gpu[block_ind*k_given*k_given+ty*k_given+threadIdx.x]=total_sum;
}



void print_one_matrix(one_matrix &matrix) {
    // Print the matrix dimensions and number of blocks
    cout << "Rows: " << matrix.rows << ", Cols: " << matrix.cols << ", Blocks: " << matrix.blocks << endl;
    
    // Iterate over the map and print the block coordinates and the block data
    for (auto &kv : matrix.mat) {
        int row = kv.first.first;
        int col = kv.first.second;
        vector<vector<uint64_t>> block = kv.second;
        
        // Print the block coordinates
        cout << "Block (" << row << ", " << col << "):" << endl;
        
        // Print the block data
        for (auto &row_data : block) {
            for (auto &elem : row_data) {
                cout << elem << " ";
            }
            cout << endl;
        }
    }
}
struct pair_hash {
    size_t operator()(pair<int,int> const &p) const noexcept {
        return ((uint64_t)p.first << 32) ^ (uint32_t)p.second;
    }
};
one_matrix helper(one_matrix &a,one_matrix&b,int &k_given,uint64_t* large_arr,uint64_t* out_arr,int *key_to_elem,int *key_to_elem_prefix,uint64_t* large_arr_gpu,uint64_t* out_arr_gpu,int *key_to_elem_gpu,int *key_to_elem_prefix_gpu){
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
     auto start_time = std::chrono::high_resolution_clock::now();
    int total_m1 = a.mat.size();
    uint64_t *A_buf = new uint64_t[(size_t)total_m1 * k_given * k_given];
    unordered_map<pair<int,int>,int,pair_hash> A_idx;
    A_idx.reserve(total_m1);

    { int p = 0;
      for (auto &kv : a.mat) {
        A_idx[kv.first] = p;
        uint64_t* dst = A_buf + (size_t)p * k_given * k_given;
        // copy each row contiguously:
        for (int r = 0; r < k_given; ++r) {
          memcpy(dst + (size_t)r * k_given,
                 kv.second[r].data(),
                 k_given * sizeof(uint64_t));
        }
        ++p;
      }
    }

    // 2) Flatten all B‐blocks similarly:
    int total_m2 = b.mat.size();
    uint64_t *B_buf = new uint64_t[(size_t)total_m2 * k_given * k_given];
    unordered_map<pair<int,int>,int,pair_hash> B_idx;
    B_idx.reserve(total_m2);

    { int p = 0;
      for (auto &kv : b.mat) {
        B_idx[kv.first] = p;
        uint64_t* dst = B_buf + (size_t)p * k_given * k_given;
        for (int r = 0; r < k_given; ++r) {
          memcpy(dst + (size_t)r * k_given,
                 kv.second[r].data(),
                 k_given * sizeof(uint64_t));
        }
        ++p;
      }
    }

    // 3) Build the “d‐map” (i,k) → list of j’s, exactly as before:
    unordered_map<int, vector<int>> m2_index;
    m2_index.reserve(b.mat.size());
    for (auto &kv : b.mat) {
        m2_index[kv.first.first].push_back(kv.first.second);
    }

    unordered_map<pair<int,int>, vector<int>, pair_hash> d;
    d.reserve(a.mat.size());
    for (auto &kv : a.mat) {
        int i = kv.first.first, j = kv.first.second;
        auto it = m2_index.find(j);
        if (it == m2_index.end()) continue;
        for (int k : it->second) {
            d[{i,k}].push_back(j);
        }
    }


    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
 //   cout <<rank<<" time taken11 " << duration.count() << " seconds" << endl;
    start_time=end_time;
  // print_one_matrix(a);
  // print_one_matrix(b);
       // int blocks_a=0;
        int total_keys = d.size();
     using MapType = decltype(d);
   using ValType = typename MapType::value_type;     // this is std::pair<const pair<int,int>, vector<int>>
   ValType **int_to_it = new ValType*[total_keys];
   {
     size_t index = 0;
     for (auto map_it = d.begin(); map_it != d.end(); ++map_it, ++index) {
       int_to_it[index] = &*map_it;  // &*iterator is a pointer to the actual Map::value_type
     }
   }
         end_time= std::chrono::high_resolution_clock::now();
     duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
  //  cout <<rank<<" time taken122 " << duration.count() << " seconds" << endl;
    start_time=end_time;
    int num_rounds = (total_keys + small_size - 1) / small_size;
    map<pair<int,int>,vector<vector<uint64_t>>> res;
    for (int round = 0; round < num_rounds; ++round) {
      int start = round * small_size;
      int end   = min(start + small_size, total_keys);

      int sum_pairs = 0;
      int iters     = 0;
        for (int idx = start; idx < end; ++idx) {
        auto &entry = *int_to_it[idx];
        auto  ik    = entry.first;     // (i,k)
        auto &js    = entry.second;    // list of j’s
        int offset  = sum_pairs * 2 * k_given * k_given;

        // **two memcpy calls per “j”**:
        for (int jn = 0; jn < (int)js.size(); ++jn) {
          int  j    = js[jn];
          int  aoff = A_idx[{ ik.first,  j }];
          int  boff = B_idx[{ j,         ik.second }];
          uint64_t *srcA = A_buf + (size_t)aoff * k_given * k_given;
          uint64_t *srcB = B_buf + (size_t)boff * k_given * k_given;

          // copy A‐block
          memcpy( large_arr + offset + (size_t)jn*2*k_given*k_given,
                  srcA,
                  (size_t)k_given * k_given * sizeof(uint64_t) );

          // copy B‐block immediately after
          memcpy( large_arr + offset + (size_t)jn*2*k_given*k_given + k_given*k_given,
                  srcB,
                  (size_t)k_given * k_given * sizeof(uint64_t) );
        }

        key_to_elem_prefix[idx - start] = sum_pairs;
        key_to_elem       [idx - start] = js.size();
        sum_pairs += js.size();
        ++iters;
      }
      double x = (sum_pairs*2*k_given*k_given)/1000000000.0;

  // cout<<x<<" "<<total_keys<<" "<<round<<endl;
            end_time = std::chrono::high_resolution_clock::now();
     duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
  //  cout <<rank<<" time taken2 " << duration.count() << " seconds" << endl;
    start_time =end_time;
      // 6) Copy to device & launch kernel exactly as before:
      cudaMemcpy( large_arr_gpu,
                  large_arr,
                  (size_t)sum_pairs * 2*k_given*k_given * sizeof(uint64_t),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( key_to_elem_prefix_gpu,
                  key_to_elem_prefix,
                  iters * sizeof(int),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( key_to_elem_gpu,
                  key_to_elem,
                  iters * sizeof(int),
                  cudaMemcpyHostToDevice );
            end_time = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
          //      cout <<rank<<" time taken222 " << duration.count() << " seconds" << endl;
                start_time =end_time;
      dim3 grid(min(small_size, total_keys - round * small_size)), block(k_given,k_given);
      matrix_multiplyKernel<<<grid,block>>>(large_arr_gpu,
                                             key_to_elem_gpu,
                                             key_to_elem_prefix_gpu,
                                             k_given,
                                             out_arr_gpu);
      cudaMemcpy( out_arr,
                  out_arr_gpu,
                  (size_t)iters * k_given * k_given * sizeof(uint64_t),
                  cudaMemcpyDeviceToHost );
      cudaDeviceSynchronize();
     end_time = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
           //     cout <<rank<<" time taken3 " << duration.count() << " seconds" << endl;
                start_time =end_time;
      // 7) Unpack results into your `res` map:
      for (int idx = start; idx < end; ++idx) {
        auto &entry = *int_to_it[idx];
        vector<vector<uint64_t>> block(k_given, vector<uint64_t>(k_given));
        uint64_t *src = out_arr + (size_t)(idx-start) * k_given * k_given;
        for (int r = 0; r < k_given; ++r) {
          memcpy(block[r].data(),
                 src + (size_t)r*k_given,
                 k_given * sizeof(uint64_t));
        }
        res[entry.first] = std::move(block);
      }
    }
     end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
 //   cout <<rank<<" time taken4 " << duration.count() << " seconds" << endl;
    start_time=end_time;
    // 8) Clean up and return
    delete[] A_buf;
    delete[] B_buf;
    delete[] int_to_it;

    one_matrix C;
    C.rows   = a.rows;
    C.cols   = b.cols;
    C.blocks = res.size();
    C.mat    = std::move(res);
    return C;
}
void helper2(int &end_ind,int &start_ind,one_matrix *arr,int &k_given,uint64_t* large_arr,uint64_t* out_arr,int *key_to_elem,int *key_to_elem_prefix,uint64_t* large_arr_gpu,uint64_t* out_arr_gpu,int *key_to_elem_gpu,int *key_to_elem_prefix_gpu){
    int len=(end_ind-start_ind+1);
    
     while(len>1){
         int new_size=len/2;
         if(len%2==1){new_size+=1;}
         one_matrix* new_arr =new one_matrix[new_size];
      //  cout<<len<<" "<<rank<<" "<<size<<endl;
       
                int to_this_val =len-2;
                for(int ind=start_ind;ind<=to_this_val+start_ind;ind+=2){
                 //   if(rank==size-1){
                 //       cout<<ind<<" dfd"<<endl;
                 //   }
                                         cout<<"multiplying "<<ind<<" "<<ind+1<<endl;
                        one_matrix &a =arr[ind];
                        one_matrix &b=arr[ind+1];
                        //cout<<ind<<endl;
                        one_matrix c =helper(a,b,k_given,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);
                       
                         int dst =(ind-start_ind)/2;
                         new_arr[dst]=c;
                        
                        
                   

                }
               
        if(len%2==0){
            len=len/2;
        }
        else{
            new_arr[new_size-1]=arr[start_ind+len-1];
            len=len/2+1;
        }
        for(int i=start_ind;i<start_ind+new_size;i++){
            arr[i]=new_arr[i-start_ind];
        }
         delete[] new_arr;
    }
}
void extract(int &start_ind, int &end_ind,one_matrix *arr,string &folder_path,int &k_given,uint64_t* large_arr,uint64_t* out_arr,int *key_to_elem,int *key_to_elem_prefix,uint64_t* large_arr_gpu,uint64_t* out_arr_gpu,int *key_to_elem_gpu,int *key_to_elem_prefix_gpu){

 //   const uint64_t MAX = 18446744073709551615ULL;
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    #pragma omp parallel num_threads(16)
    {
        #pragma omp single 
        {
            for(int i=start_ind+1;i<end_ind+2;i++){
                #pragma omp task
                {
                    map<pair<int,int>,vector<vector<uint64_t>>>m;
                    string original ="/matrix";
                    int p=i;
                    original+=to_string(p);
                    ifstream mat_file(folder_path + original);
                    if (!mat_file) {
                        cerr << "Cannot open size file!" << endl;
                       // return ;
                    }
                    
                   // cout<<folder_path+original<<endl;
                    int rows,cols;
                    mat_file>>rows>>cols;
                   // cout<<rows<<" "<<cols<<endl;
                   
                //     cout<<start_ind<<endl;
                //    if(rank==size-1){
                //         cout<<folder_path+original<<endl;
                //         cout<<rows<<" "<<cols<<endl;
                //     }
                   
                    int blocks ;
                    mat_file>>blocks;
                    for(int j=0;j<blocks;j++){
                        int row1,col1;
                        mat_file>>row1>>col1;
                        vector<vector<uint64_t>>v;
                        int k1=k_given;
                        
                        int k3=k_given;
                        
                        for(int l=0;l<k1;l++){
                            vector<uint64_t>v1;
                            for(int o=0;o<k3;o++){
                                uint64_t inp;
                                mat_file>>inp;
                                v1.push_back(inp);
                            }
                            v.push_back(v1);
                        }
                        

                        m[{row1,col1}]=v;
                    }
                   
                     //   cout<<rank<<endl;
                        
                    arr[i-1].rows=rows;
                    arr[i-1].cols =cols;
                    arr[i-1].blocks =blocks;
                    arr[i-1].mat =m;
                  //  print_one_matrix(arr[i-1]);
                    
                }
            }
        }
    }
     helper2(end_ind,start_ind,arr,k_given,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);


};
int main(int argc ,char* argv[]){
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc,&argv);
    string folder_path = argv[1];
   // cout<<folder_path<<endl;
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int N,k;
    {
        ifstream size_file(folder_path +"/size");
        if (!size_file) {
         //   cout<<"hi"<<endl;
            cerr << "Cannot open size file!" << endl;
            return 1;
        }
        size_file >>N>>k;
    }
    int k_given=k;
    uint64_t* large_arr=new uint64_t[BIG_SIZE];
    uint64_t *out_arr= new uint64_t[small_size*k_given*k_given];
    int * key_to_elem=new int[small_size];
    int * key_to_elem_prefix=new int[small_size];
    uint64_t* large_arr_gpu;
    CUDA_CHECK(cudaMalloc(&large_arr_gpu, BIG_SIZE*sizeof(uint64_t)));
    // cudaMalloc(&large_arr_gpu,100000000*sizeof(uint64_t));
    uint64_t *out_arr_gpu;
    // cudaMalloc(&out_arr_gpu,1000*k_given*k_given*sizeof(uint64_t));
    int * key_to_elem_gpu;
    // cudaMalloc(&key_to_elem_gpu,1000*sizeof(int));
    int * key_to_elem_prefix_gpu;
    // cudaMalloc(&key_to_elem_prefix_gpu,1000*sizeof(int));
    CUDA_CHECK(cudaMalloc(&out_arr_gpu,           small_size*k_given*k_given*sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&key_to_elem_gpu,       small_size*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&key_to_elem_prefix_gpu,small_size*sizeof(int)));
    one_matrix* arr =new one_matrix[N];
    int one_processing ;
    one_processing=N/size;
    if(one_processing!=0){
    if(rank!=size-1){
        int start_ind = (rank)*one_processing;
        int end_ind = (rank+1)*one_processing-1;
     //   cout<<rank<<" "<<start_ind<<" "<<end_ind<<endl;
             extract(start_ind,end_ind,arr,folder_path,k,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);

     //   print_one_matrix(arr[0]);
    }
    if(rank ==size-1){
        int start_ind = (rank)*one_processing;
        int end_ind = N-1;
    //    cout<<rank<<" "<<start_ind<<" "<<end_ind<<endl;
         extract(start_ind,end_ind,arr,folder_path,k,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);

       // print_one_matrix(arr[4]);
    }
   // cout<<"\n\n\n\n\n\n\n\n\n\n\n\n\n\n"<<endl;
    int start_ind = (rank)*one_processing;
    
    one_matrix *arr1 = nullptr;
    if (rank == 0) {
        arr1 = new one_matrix[size];
    }

    // Flatten/send header, keys, and values
    {
        const int CHUNK_KEYS = 1024 * 256;    // tune as you like
        const int CHUNK_VALS = 1024 * 4096;
        one_matrix &M = arr[start_ind];

        if (rank != 0) {
            // 1) header = [rows, cols, blocks]
            int header[3] = { M.rows, M.cols, M.blocks };
            MPI_Send(header, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // 2) keys: 2 ints per block
            int B = M.blocks;
            vector<int> keys(2 * B);
            { int b = 0;
            for (auto &kv : M.mat) {
                keys[2*b    ] = kv.first.first;
                keys[2*b + 1] = kv.first.second;
                ++b;
            }
            }
            for (int off = 0; off < (int)keys.size(); off += CHUNK_KEYS) {
            int cnt = min(CHUNK_KEYS, (int)keys.size() - off);
            MPI_Send(&keys[off], cnt, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }

            // 3) values: B * k_given*k_given uint64_t’s
            int KK = k * k;  // your block‐size
            vector<uint64_t> vals((size_t)B * KK);
            { int b = 0;
            for (auto &kv : M.mat) {
                auto &blk = kv.second;
                for (int r = 0; r < k; r++)
                for (int c = 0; c < k; c++)
                    vals[(size_t)b*KK + r*k + c] = blk[r][c];
                ++b;
            }
            }
            for (int off = 0; off < (int)vals.size(); off += CHUNK_VALS) {
            int cnt = min(CHUNK_VALS, (int)vals.size() - off);
            MPI_Send(&vals[off], cnt, MPI_UNSIGNED_LONG_LONG, 0, 2, MPI_COMM_WORLD);
            }

        } else {
            // rank 0: first copy its own matrix
            arr1[0] = M;

            // then receive from ranks 1..size-1
            for (int src = 1; src < size; ++src) {
                one_matrix &R = arr1[src];

                // 1) header
                int header[3];
                MPI_Recv(header, 3, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                R.rows   = header[0];
                R.cols   = header[1];
                R.blocks = header[2];

                // 2) keys
                int B = R.blocks;
                vector<int> keys(2 * B);
                for (int off = 0; off < 2*B; off += CHUNK_KEYS) {
                int cnt = min(CHUNK_KEYS, 2*B - off);
                MPI_Recv(&keys[off], cnt, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // 3) values
                int KK = k * k;
                vector<uint64_t> vals((size_t)B * KK);
                for (int off = 0; off < (int)vals.size(); off += CHUNK_VALS) {
                int cnt = min(CHUNK_VALS, (int)vals.size() - off);
                MPI_Recv(&vals[off], cnt, MPI_UNSIGNED_LONG_LONG, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // Rebuild the sparse‐map
                R.mat.clear();
                for (int b = 0; b < B; b++) {
                    int bi = keys[2*b], bj = keys[2*b+1];
                    vector<vector<uint64_t>> blk(k, vector<uint64_t>(k));
                    for (int r = 0; r < k; r++) {
                        for (int c = 0; c < k; c++) {
                            blk[r][c] = vals[(size_t)b*KK + r*k + c];
                        }
                    }
                    R.mat[{bi,bj}] = std::move(blk);
                }
            }
        }
    }

    // synchronize and now rank 0 has arr1[0..size-1]
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
    //    for(int i=0;i<size;i++){
    //        cout<<i<<endl;
    //        print_one_matrix(arr1[i]);
    //    }
      //  const uint64_t MAX = 18446744073709551615ULL;
        int k_given =k;





        start_ind=0;
        int end_ind=size- 1;
        helper2(end_ind,start_ind,arr1,k_given,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);

        map<pair<int,int>,vector<vector<uint64_t>>>&ans=arr1[0].mat;
        int num_blocks =arr1[0].blocks;
        int R=arr1[0].rows;
        int C=arr1[0].cols;
        for(auto &it:ans){
            vector<vector<uint64_t>>&v=it.second;
            bool is_zero=true;
            for(int i=0;i<k;i++){
                for(int j=0;j<k;j++){
                    if (v[i][j]>0){
                        is_zero=false;
                    }
                }
            }
            if (is_zero==true){
                num_blocks--;
               ans.erase(it.first);
            }
            
        }
       // print_one_matrix(arr1[0]);
     //   cout<<arr1[0].rows<<" "<<arr1[0].cols<<" "<<arr1[0].blocks<<endl;
        ofstream out_file("matrix");
        out_file<<R<<" "<<C<<endl;
        out_file<<num_blocks<<endl;
        for(auto &it:ans){
            out_file<<it.first.first<<" "<<it.first.second<<endl;
            for(auto &i:it.second){
                for (int j=0;j<i.size()-1;j++){
                    out_file<<i[j]<<" ";
                }
                out_file<<i[i.size()-1];
                out_file<<endl;
            }
        }
        out_file.close();
        delete[] arr1;
    }
    }
    else{if(rank==0){
        int start_ind = (rank)*one_processing;
        int end_ind = N-1;
        extract(start_ind,end_ind,arr,folder_path,k,large_arr,out_arr,key_to_elem,key_to_elem_prefix,large_arr_gpu,out_arr_gpu,key_to_elem_gpu,key_to_elem_prefix_gpu);

        map<pair<int,int>,vector<vector<uint64_t>>>&ans=arr[0].mat;
        int num_blocks =arr[0].blocks;
        int R=arr[0].rows;
        int C=arr[0].cols;
        for(auto &it:ans){
            vector<vector<uint64_t>>&v=it.second;
            bool is_zero=true;
            for(int i=0;i<k;i++){
                for(int j=0;j<k;j++){
                    if (v[i][j]>0){
                        is_zero=false;
                    }
                }
            }
            if (is_zero==true){
                num_blocks--;
                ans.erase(it.first);
            }
            // int r=it.first.first;
            // int c=it.first.second;
            // if(r+k>R or c+k>C){
            //      vector<vector<uint64_t>>v1;
            //      for(int i=0;i<R-r;i++){
            //         vector<uint64_t>v2;
            //         for(int j=0;j<C-c;j++){
            //             v2.push_back(v[i][j]);
            //         }
            //         v1.push_back(v2);
            //      }
            //      ans[it.first]=v1;
            // }
        }
       // print_one_matrix(arr1[0]);
     //   cout<<arr1[0].rows<<" "<<arr1[0].cols<<" "<<arr1[0].blocks<<endl;
        ofstream out_file("matrix");
        out_file<<R<<" "<<C<<endl;
        out_file<<num_blocks<<endl;
        for(auto &it:ans){
            out_file<<it.first.first<<" "<<it.first.second<<endl;
            for(auto &i:it.second){
                for (int j=0;j<i.size()-1;j++){
                    out_file<<i[j]<<" ";
                }
                out_file<<i[i.size()-1];
                out_file<<endl;
            }
        }
        out_file.close();
    }
    }
    delete []arr;
    delete [] large_arr;
        delete [] out_arr;
        delete [] key_to_elem;
        delete [] key_to_elem_prefix;
        cudaFree(large_arr_gpu);
        cudaFree(out_arr_gpu);
        cudaFree(key_to_elem_gpu);
        cudaFree(key_to_elem_prefix_gpu);
    MPI_Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    cout << "time taken " << duration.count() << " seconds" << endl;

    return 0;
}