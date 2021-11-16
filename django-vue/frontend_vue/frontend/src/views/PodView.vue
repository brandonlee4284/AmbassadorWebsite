<template>
  <div class="pod-view">
    <div v-for="pod in APIData" :key="pod.id" class="col-md-4">

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
    }
    
  }
</script>

<style scoped>

</style>