<template>
  <div id="activities">
    <div class="album py-5 bg-light">
    <div class="container">
        <h1 style="font-family:Gabriola"><i class="fas fa-gamepad" style="font-size:32px"></i> Activities</h1>
        <div v-for="schedule in APIData" :key="schedule.id" class="col-md-6">

              <hr>
              <h4 class=""><p class="">{{schedule.activity}}</p></h4>
              <p class="">{{schedule.activity_description}}</p>

        </div>
        
    </div>
    </div>
  </div>
</template>

<script>
import { getAPI } from '../axios-api'
import { mapState } from 'vuex'
export default {
  name: "Activities",
  mounted() {
    document.title = 'Activities'
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

<style>
</style>

