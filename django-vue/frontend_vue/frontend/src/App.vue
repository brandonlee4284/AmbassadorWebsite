<template>
  <div id="app">
    
    <Navbar />
    <div :style="{ 'margin-left': sidebarWidth }">
    <router-view />
    

    <footer class="text-center text-lg-start bg-light text-muted" v-if="accessToken!=null">
      <section class="d-flex justify-content-center justify-content-lg-between p-4 border-bottom">
        <div>
          <a href="https://www.facebook.com/MountainViewSpartans/" target="_blank" class="me-4 text-reset">
            <i class="fab fa-facebook-f"></i>
          </a>
          <a href="https://twitter.com/MvhsSpartans" target="_blank" class="me-4 text-reset">
            <i class="fab fa-twitter"></i>
          </a>
          <a href="https://www.instagram.com/mvambassadors/?hl=en" target="_blank" class="me-4 text-reset">
            <i class="fab fa-instagram"></i>
          </a>
          <a href="#" target="_blank" class="me-4 text-reset">
            <i class="fab fa-youtube"></i>
          </a>
        </div>
    </section>

  <section class="">
    <div class="container text-center text-md-start mt-5">
      <div class="row mt-3">
        <div class="col-md-3 col-lg-4 col-xl-3 mx-auto mb-4">
          <h6 class="text-uppercase fw-bold mb-4">
            <i></i>Mountain View Highschool
          </h6>
          <p><i class="fas fa-globe"></i>
            School Website:
            <a class="text-reset fw-bold" href="https://mvhs.mvla.net/" target="_blank">mvla.net/mvhs</a>
            <br><br>
            
            <a href="https://mvhs.mvla.net/" target="_blank">
              <img src="https://mvhs.mvla.net/images/logo.png" alt="mvhs-logo" class="toright" width = "240" height = "140">
            </a>
            
          </p>
          </div>

          <div class="col-md-2 col-lg-2 col-xl-2 mx-auto mb-4">
            <h6 class="text-uppercase fw-bold mb-4">
              Quick Links
            </h6>
            <p>
              <a href="/" class="text-reset">Home</a>
            </p>
            <p>
              <a href="/pods" class="text-reset">View Pods</a>
            </p>
            <p>
              <a href="/schedule" class="text-reset">Day Schedule</a>
            </p>
            <p>
              <a href="/resources" class="text-reset">Help</a>
            </p>
          </div>

          <div class="col-md-3 col-lg-2 col-xl-2 mx-auto mb-4">
            <h6 class="text-uppercase fw-bold mb-4">
              Other links
            </h6>
            <p>
              <a href="http://127.0.0.1:8000/admin/" target="_blank" class="text-reset">Admin Login</a>
            </p>
            <p>
              <a><router-link :to = "{ name:'developers' }" class="text-reset">Developers</router-link></a>
            </p>
            <p>
              <a><router-link :to = "{ name:'log-out' }" style="color:red">Logout</router-link></a>
              
            </p>
            
            
          </div>

          <div class="col-md-4 col-lg-3 col-xl-3 mx-auto mb-md-0 mb-4">
            
            <p><i class="fas fa-home me-6"></i> 3535 Truman Ave, Mountain View, CA 94040</p>
            <p>
              <i class="fas fa-envelope me-3"></i>
              mvhsadmin@mvla.net
            </p>
            <p><i class="fas fa-phone me-3"></i> (650) 940-4600</p>
            <p><i class="fas fa-fax"></i> (650) 961-6349 (fax)</p> 

          </div>
        </div>
      </div>
    </section>

    <div class="text-center p-4" style="background-color: rgba(0, 0, 0, 0.05);">
      Copyright © 2021-2022
      <br><br>
      Website Developed By Alex Lio and Brandon Lee
      
    </div>
  </footer>
</div>


  </div>
</template>

<script>
import { mapState } from 'vuex'
import Navbar from './components/Navbar'
import { sidebarWidth } from '@/components/state'
export default {
  name: 'App',
  components: {
      Navbar,
  },
  
  computed: mapState(['accessToken']),
  
  setup() {
    return { sidebarWidth }
    
  },
  
  
  
}
</script>


<style>
  body {
    margin: 0;
    padding: 0;
    background-color:#f8f8f8!important;
  }
  .toright{
    position: relative;
    right: 30px;
  }
  #app {
    font-family: Helvetica, Arial, sans-serif;
    color: #2c3e50;
  }
  #nav {
    padding: 30px;
  }
  #nav a {
    font-weight: bold;
    color: #2c3e50;
  }
  #nav a.router-link-exact-active {
    color: #f8f8f8;
  }
</style>
