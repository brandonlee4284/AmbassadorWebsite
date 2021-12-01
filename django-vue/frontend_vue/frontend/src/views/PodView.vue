<template>
  <div class="pod-view">
    <div class="album py-5 bg-light">
      <div class="container">
        <div class="row">
          <!--
            if a Pod (i) button is pressed
              show pod in APIData.slice(i,i+1)
          -->
          <div v-for="pod in APIData.slice(pod_number-1,pod_number)" :key="pod.id" class="">
            <h1 class="center" style="margin:2vw 0 6vw 0">{{pod.pod_group_number}}</h1>
            <h5><b>Pod Leader(s):</b> {{pod.pod_leader}}</h5>
            <p><b>Room:</b> {{pod.pod_room_number}}</p>
            <p><b>Members:</b> {{pod.pod_group_members}}</p>
            <p><b>Additional Notes:</b> {{pod.additional_notes}}</p>
            <br><br><br><br><br><br>
          </div>
        </div>
      </div>
    </div>
  </div>

</template>

<script>
import { getAPI } from '../axios-api'
import { mapState } from 'vuex'
import { pod_number } from './Pods.vue'
export default {
  name: 'PodView',
  setup() {
    window.scrollTo(0,0);
    return { pod_number }
    
  },
  mounted() {
  document.title = 'Pod View'
  },
  computed: mapState(['APIData']),
  created () {
      getAPI.get('/pod/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
        .then(response => {
          this.$store.state.APIData = response.data
          this.pod_number = pod_number
          //console.log(pod_number)
        })
        .catch(err => {
          console.log(err)
        })
  },
  method: {
    getPod(){
      return pod_number;
    },
  },

  
}
</script>

<style scoped>
.center{
  text-align: center;
}


</style>