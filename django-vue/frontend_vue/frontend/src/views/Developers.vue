<template>
  <div id="developers" class="fontMode">
    <div class="album py-5">
    <div class="container">
      <h1 style="font-family:Cambria"><i class="fas fa-users-cog" style="font-size:32px"></i> <b>Developers</b></h1>
    <div>
      <br><br>
      <p>This website was designed and developed by Mountain View High School students 
        of Class 2022. Our mission/goal for this project was to create an accessible and easy to use 
        application that would allow pod leaders to access any information or logistics needed for freshman orientation.
        Because of its intent for guidance before and during freshman orientation, we incorporated several 
        features such as pages for important events, day-to-day operations of orientation itself, pod groups and 
        their corresponding demographic, and other resources provided by admin. Additionally when designing and 
        building the website we emphasized the importance for a dynamic website to accommodate the compatibility across 
        a variety of devices whether that be a computer or mobile device. 
        <br><br> 
        If interested in contributing to this project consider joining the MVHS Computer Science club to help maintain 
        a bug-free and improved experience for future pod leaders!
        <br><br><br>
        <i>Brandon Lee and Alex Lio</i>
        <br>
        <i>Class 2022</i>
        <br>
        <i>Website Developers</i>
      </p>


        <br>
    </div>          
    </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Developers",
  mounted() {
    document.title = 'Developers'
  },
}
</script>

<style>
.size{
  font-size: 30px;
}
.fontMode{
  color: var(--text-primary-color);
}


</style>