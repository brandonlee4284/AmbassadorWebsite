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
            <h1 class="" style="margin:4vw 0 0 0;font-family:Cambria">{{pod.pod_group_number}}</h1>
            <p>Room: {{pod.pod_room_number}}</p>
            <p>Demographic: <b>Gender Ratio:</b> {{pod.gender_ratio}} | <b>School Ratio:</b> {{pod.school_ratio}} | <b>Total Students:</b> {{pod.total_students}}</p>
            
            <table id="students">
              <tr>
                <th>Pod Leader(s)</th>
                <th>Members</th>
                
              </tr>
              <tr>
                <td class="center">{{pod.pod_leader}}</td>
                <div v-for="pod in APIData.slice(pod_number-1,pod_number)" :key="pod.id" class="">
                  <td>{{pod.pod_group_members}}</td>
                </div>
                
              </tr>
            </table>
            <br>
            <p><b>Additional Notes:</b></p>
            <p>{{pod.additional_notes}}</p>          
            <br><br>
            
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
  document.title = "Pod " + pod_number
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
#students {
  font-family: Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

#students td, #students th {
  border: 1px solid #ddd;
  padding: 8px;
}

#students tr:nth-child(even){background-color: #f2f2f2;}

#students tr:hover {background-color: #ddd;}

#students th {
  padding: 12px 20px;
  text-align: center;
  background-color: #FFD700;
  color: black;
}

</style>