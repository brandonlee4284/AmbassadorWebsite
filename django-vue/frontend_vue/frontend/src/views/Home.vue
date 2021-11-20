<template>
    <div id="home" class="">
       <div class="album py-5 bg-light">
        <div class="container">
            <h1><i class="fas fa-calendar-alt" style="font-size:32px"></i> Upcoming Events</h1>
            <!--<div v-for="homeImage in APIData1" :key="homeImage.id" class="col-md-6">
              

            </div>-->

            <!-- -->
            <a href="https://mvhs.mvla.net/Student-Activities/2022-Senior-Year/index.html" target="_blank">
                <img src="https://mvhs.mvla.net/pictures/Seniors.jpg" alt="senior-annoucement" class="responsive" width="500" height="300">
            </a>
            <br>
            <div class="dropdown">
              <button class="dropbtn">Filter</button>
              <div class="dropdown-content">
                <a href="#">Latest to Oldest</a>
                <a href="#">Oldest to Latest</a>
              </div>
            </div>
            
            <div v-for="home in APIData" :key="home.id" class="col-md-6">

              <hr>
              <h4 class=""><a>
                {{home.annoucement_title}}
              </a></h4>
              <a class="text-secondary" href="https://calendar.google.com/calendar/r/eventedit?text=My+Custom+Event&dates=20180512T230000Z/20180513T030000Z&details=For+details,+link+here:+https://example.com/tickets-43251101208&location=Garage+Boston+-+20+Linden+Street+-+Allston,+MA+02134" target="_blank">{{home.date_time}}</a>
              <p class="">{{home.description}}</p>

            </div>
          </div>
        </div>
       </div>
    
</template>

<script>
import { getAPI } from '../axios-api'
import { mapState } from 'vuex'

export default {
  name: 'Home',
  components: {
    
  },
  methods:{
    
      
  },
  
  
  mounted() {
    document.title = 'Home'
  },
  computed: mapState(['APIData']),  
  created () {
    getAPI.get('/home/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
      .then(response => {
        this.$store.state.APIData = response.data.slice().reverse()
      })
      .catch(err => {
        console.log(err)
      })

     
     getAPI.get('/homeImage/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
      .then(response => {
        this.$store.state.APIData1 = response.data
        
      })
      .catch(err => {
        console.log(err)
      })
     


    },
   


}
</script>


<style scoped>
.dropbtn {
  background-color: black;
  color: white;
  padding: 7px;
  font-size: 16px;
  border: none;
  width: 100px;
  margin: 10px 0px;
}

.dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-content {
  display: none;
  position: absolute;
  background-color: #f1f1f1;
  min-width: 160px;
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

.responsive {
  max-width: 100%;
  height: auto;

}

img:hover{
  box-shadow: 0 50px 50px 0 rgba(0,0,0,0.2);
}
img{
  transition: 0.3s;
}



 
</style>