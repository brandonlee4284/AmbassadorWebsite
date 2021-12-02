<template>
  <div id="resources">
    <div class="album py-5 bg-light">
    <div class="container">
        <h1 style="font-family:Gabriola"><i class="fas fa-book" style="font-size:32px"></i> Additional Resources</h1>
        <br><br><br>
        <i class="fas fa-chevron-right" style="font-size:24px"></i> 
        <i class="fas fa-chevron-right" style="font-size:24px;margin:0 10px 0 0"></i> 
        <a class="size"><router-link :to = "{ name:'activities' }" exact>Activities..</router-link></a>

        
         <div v-for="resources in APIData" :key="resources.id" class="col-md-6">
            <br><br><br>
            <i class="fas fa-chevron-right" style="font-size:24px;"></i>
            <i class="fas fa-chevron-right" style="font-size:24px;margin:0 10px 0 0"></i> 
            <a class="size" :href="resources.link" target="_blank"> {{resources.resource_name}}</a>
         </div>


         <br><br><br>
          <i class="fas fa-chevron-right" style="font-size:24px"></i>
          <i class="fas fa-chevron-right" style="font-size:24px;margin:0 10px 0 0"></i> 
          <a class="size"><router-link :to = "{ name:'developers' }" exact>Developers..</router-link></a>
         <br><br><br><br>

    </div>
    </div>
  </div>
</template>

<script>
import { getAPI } from '../axios-api'
import { mapState } from 'vuex'
export default {
  name: "Resources",
  mounted() {
    document.title = 'Resources'
  },
  computed: mapState(['APIData']),  
  created () {
    getAPI.get('/resources/', { headers: { Authorization: `Bearer ${this.$store.state.accessToken}` } })
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
.size{
  font-size: 30px;
}
</style>

