<template>
  <div id="resources">
    <div class="album py-5 bg-light">
    <div class="container">
        <h1 class=""><i class="fas fa-book" style="font-size:32px"></i> Additional Resources</h1>
        <br><br><br>
        <i class="fas fa-chevron-right" style="font-size:24px"></i> 
        <i class="fas fa-chevron-right" style="font-size:24px"></i> 
        <a class="size"><router-link :to = "{ name:'activities' }" exact>
         Activities..</router-link></a>

        <br><br><br>
        <i class="fas fa-chevron-right" style="font-size:24px"></i>
        <i class="fas fa-chevron-right" style="font-size:24px"></i> 
        <a class="size"><router-link :to = "{ name:'developers' }" exact>
         Developers..</router-link></a>
         
         <br><br><br>

    </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Resources",
  mounted() {
    document.title = 'Resources'
  },
}
</script>

<style>
.size{
  font-size: 30px;
}
</style>

