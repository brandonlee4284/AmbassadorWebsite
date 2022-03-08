<template>
  <div class="outer-mode"><br><br>
  <div class="container text-dark">
      <div class="col-md-5 p-3 login justify-content-md-center mode"><hr>
        <img src="https://mvhs.mvla.net/images/logo.png" alt="mvhs-logo" class="i">
        <br><br>
        <h1 class="h3 mb-3 font-weight-normal text-center" style="font-family:Cambria"><b>Ambassador Login</b></h1>
        <br>
        
        <form v-on:submit.prevent="login">
          <div class="form-group">
            <input type="text" name="username" id="user" v-model="username" class="form-control" placeholder="Username">
          </div>
          <br><br>
          <div class="form-group">
            <input type="password" name="password" id="pass" v-model="password" class="form-control" placeholder="Password">
          </div>
          <br>
          <p v-if="incorrectAuth" class="incorrect">Invalid password or username</p>
          <button type="submit" class="btn btn-md btn-primary btn-block">Login</button>
        </form>

        
        <br><br><br>
        <hr>
      </div>
      <br><br><br><br><br><br><br><br><br>
    </div>
  </div>
</template>

<script>
export default {
  
    name: 'login',
    data () {
      return {
        username: '',
        password: '',
        incorrectAuth: false
      }
    },
    mounted() {
    document.title = 'Login'
    },
    

    
    methods: {
      login () { 
        this.$store.dispatch('userLogin', {
          username: this.username,
          password: this.password
        })
        .then(() => {
          this.$router.push({ name: 'home' })
          //toggleSidebar()
        })
        .catch(err => {
          console.log(err)
          this.incorrectAuth = true
        })
        }
      }
  }
</script>

<style>


.login{
  background-color:white;
  transition: 0.3s;
  margin: auto;

}

.i {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
  
}

.incorrect{
  color: red;
}
.mode{
  background-color: var(--background-color-secondary);
  color: var(--text-primary-color);
}
.outer-mode{
  background-color: var(--background-color-primary);
}



</style>