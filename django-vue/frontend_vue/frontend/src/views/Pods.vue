<template>
  <div class="pods">
    <div class="album py-5 bg-light">
        <div class="container">
          <div class="row">
            <h1 style="font-family:Gabriola"><i class="fas fa-users"></i> Assigned Pods</h1>
            <div>
              <form class="search">
                <input type="text" placeholder="Search Pod Group.." name="search" style="position:relative;left:-10px">
                <button type="">Search</button>
              </form>
              
            </div>
            
            
            <div>
              <div class="dropdown">
                <button class="dropbtn">Sort By...</button>
                <div class="dropdown-content">
                  <a>Pod Group Number</a>
                  <a>Pod Leader (Alphabetical)</a>
                </div>
              </div>
            </div>
            
            <div v-for="pod in APIData" :key="pod.id" class="col-md-4">
              <div class="card mb-4 box-shadow">
              <router-link :to = "{ name:'pod-view' }" exact>
                <img class="card-img-top" src="https://images.unsplash.com/photo-1579546929518-9e396f3cc809?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8&w=1000&q=80" alt="Card image cap" @click="getNumber(pod.pod_group_number)">
              </router-link>
                <div class="card-body">
                    <h4><a><router-link :to = "{ name:'pod-view' }" exact class="text-secondary" @click="getNumber(pod.pod_group_number)">
                      {{pod.pod_group_number}}
                    </router-link></a></h4>
                    <p class="card-text">{{pod.pod_leader}}</p>
                    <div class="d-flex justify-content-between align-items-center">

                    
                    <div class="btn-group" >
                     <a class="btn btn-sm btn-outline-primary" role="button" aria-pressed="true">
                        <router-link :to = "{ name:'pod-view' }" exact :id="pod.pod_group_number" @click="getNumber(pod.pod_group_number)">
                          Members
                        </router-link>
                      </a>                 
                    </div>


                    <small class="text-muted">Room {{pod.pod_room_number}}</small>
                  </div>
                </div>
              </div>
            </div>
            <a href="#" @click="scrollToTop()"><i class="fas fa-arrow-circle-up" /> Back to top</a>
          </div>
        </div>
    </div>
  </div>
</template>

<script>
  import { getAPI } from '../axios-api'
  import { mapState } from 'vuex'
  export var pod_number;
  export default {
    name: 'Pods',
    mounted() {
    document.title = 'View Pods'
    },
    computed: mapState(['APIData']),
    created () {
        getAPI.get('/pod/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
          .then(response => {
            this.$store.state.APIData = response.data
          })
          .catch(err => {
            console.log(err)
          })
    },
    methods: { 
    scrollToTop() {
      window.scrollTo(0,0);
    },
    getPod(pod_group_number){
      for(let i=0;i<this.$store.state.APIData.length;i++){
        if(this.$store.state.APIData[i].pod_group_number == pod_group_number){
          console.log(this.$store.state.APIData[i])
          console.log(i+1);
          console.log(document.getElementById(pod_group_number).id);
          
          
        }
        
      }
    },
    getNumber(pod_group_number){
      var pod = document.getElementById(pod_group_number).id.substring(4, );
      pod_number = parseInt(pod);
      //console.log(pod_number)
      return pod_number;
        
    }
  
    
    
    
  }
  
  }
 
</script>

<style scoped>
.search{
  float: right;
  padding: 10px 0;

}
.card:hover {
  box-shadow: 0 50px 50px 0 rgba(0,0,0,0.2);
}
.card{
  transition: 0.3s;
}

.dropbtn {
  background-color: black;
  color: white;
  padding: 7px;
  font-size: 16px;
  border: none;
  width: 150px;
  margin: 10px 0px;
}

.dropdown {
  position: relative;
  float: right;
  
}

.dropdown-content {
  display: none;
  position: absolute;
  background-color: #f1f1f1;
  min-width: 100px;
  box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
  z-index: 1;
}

.dropdown-content a {
  color: black;
  padding: 12px 16px;
  text-decoration: none;
  display: block;
}

.dropdown-content a:hover {background-color: #ddd;}

.dropdown:hover .dropdown-content {display: block;}

.dropdown:hover .dropbtn {background-color: gray;}


</style>