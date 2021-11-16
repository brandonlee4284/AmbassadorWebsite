<template>  

  <div v-if="accessToken!=null">
    <div class="sidebar" :style="{ width: sidebarWidth }">
    <h1>
      <span v-if="collapsed" style="font-family: Copperplate;">
        <div><a><img src="https://avatars.githubusercontent.com/u/10067543?s=280&v=4" alt="mvhs-logo" width = "30" height = "30"></a></div>
      
      </span>
      <span v-else><a><img src="https://mvhs.mvla.net/images/logo.png" alt="mvhs-logo" width = "160" height = "80"></a></span>
    </h1>

    
    <SidebarLink to="/" icon="fas fa-home">Home</SidebarLink>
    <SidebarLink to="/pods" icon="fas fa-users">Pods</SidebarLink>
    <SidebarLink to="/schedule" icon="fas fa-calendar">Schedule</SidebarLink>
    <SidebarLink to="/resources" icon="fas fa-book">Resources</SidebarLink>
    <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
    <SidebarLink to="/log-out" icon="fas fa-sign-out-alt">Logout</SidebarLink>



    <div 
    v-if="$isMobile()" 
    v-on="collapsed=true"
    >
    </div>



    <div v-else>
      <span 
        class="collapse-icon"
        :class="{ 'rotate-180': collapsed }"
        @click="toggleSidebar"
      >
        <i class="fas fa-angle-double-left" />
      </span>
    </div>
    
  </div>
    


  </div>

  
</template>

<script>
import { mapState } from 'vuex'
import SidebarLink from './SidebarLink'
import { collapsed, toggleSidebar, sidebarWidth } from './state'
  export default {
    name: 'Navbar',
    computed: mapState(['accessToken']),
    props: {},
    components: { SidebarLink },
    setup() {
    return { collapsed, toggleSidebar, sidebarWidth }
  }
    
 }

</script>

<style>
:root {
  --sidebar-bg-color: #FFD700;
  --sidebar-item-hover: #ffffff62;
  --sidebar-item-active: #ffffff;
}
</style>

<style scoped>
.sidebar {
  color: black;
  background-color: var(--sidebar-bg-color);
  float: left;
  position: fixed;
  z-index: 1;
  top: 0;
  left: 0;
  bottom: 0;
  padding: 0.5em;
  transition: 0.3s ease;
  display: flex;
  flex-direction: column;
}
.sidebar h1 {
  height: 2.5em;
}
.collapse-icon {
  position: absolute;
  bottom: 0;
  padding: 0.75em;
  color: black;
  transition: 0.2s linear;
}
.rotate-180 {
  transform: rotate(180deg);
  transition: 0.2s linear;
}
</style>