<template>
  <div class="pod-view">
    <div class="album py-5 bg-light">
      <div class="container">
        <div class="row">
          <!--
            if a Pod (i) button is pressed
              show pod in APIData.slice(i,i+1)

            const element = document.getElementById('profile_title')

            // always checking if the element is clicked, if so, do alert('hello')
            element.addEventListener("click", () => {
              alert('hello');
            });


          -->
          <div v-for="pod in APIData.slice(0,1)" :key="pod.id" class="col-md-4">
            <h1>{{pod.pod_group_number}}</h1>
            <h3>{{pod.pod_leader}}</h3>
            <p>Room: {{pod.pod_room_number}}</p>
            <p>{{pod.pod_group_members}}</p>
            
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
  name: 'PodView',
  
  
  mounted() {
  document.title = 'Pod View'
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
  method: {
    
    
  },

  
}
</script>

<style scoped>


</style>