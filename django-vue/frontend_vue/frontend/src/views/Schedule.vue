<template>
  <div id="schedule">
    <br> 
    <div class="album py-5">
      <div class="container">
        <div class="row">
          <div class="card mb-1 box-shadow mode_schedule" style="color:black">
            <br><br>
            <h2 style="font-family:Cambria; font-size: 25px" class="font-mode">Schedule</h2>
            <br><br>
            <div>
              <div>
                <span class="activity-m font-mode">Activity</span>
                <span class="time time-m font-mode">Time</span>
              </div>
            </div>
            <div v-for="schedule in APIData" :key="schedule.id">
              <hr><br><br><br>
              <span class="activity-m font-mode"><a><router-link :to = "{ name:'activities' }" exact class="link-mode">{{schedule.activity}}</router-link></a></span>
              <span class="time time-m font-mode"><a>{{schedule.time_slot}}</a></span>
              <br><br><br><br>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { getAPI } from '../axios-api'
import { mapState } from 'vuex'

export default {

  name: 'Schedule',
  components: {
    
  },
  mounted() {
    document.title = 'Schedule'
  },
  computed: mapState(['APIData']),
  data(){
    return {
            
    }
  },
  methods: {
   
  },
  

  created () {
    getAPI.get('/schedule/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
      .then(response => {
        this.$store.state.APIData = response.data
      })
      .catch(err => {
        console.log(err)
      })
  }
  
  
  
}
</script>


<style scoped>

hr {
  max-width: 100vw;
}

.card {
  transition: 1s;
}


.card:hover {
  box-shadow: 0 50px 50px 0 rgba(0,0,0,0.2);
}

img {
  border-radius: 5px 5px 0 0;
}

.container {
  padding: 2px 16px;
}

h2{
  text-align: center;
}
.time{
  float: right;
}
.activity-m{
  padding-left: 5vw;
}
.time-m{
  padding-right: 5vw;
}

.mode_schedule{
  background-color: var(--background-color-pods);
  
}

.font-mode{
  color: var(--text-primary-color);
}
.link-mode{
  color: var(--background-color-links);
}
 
</style>