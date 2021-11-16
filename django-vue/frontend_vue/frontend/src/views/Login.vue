<template>
  <div class="background"><br><br><br>
  <div class="container text-dark">
    <div class="row justify-content-md-center">
      <div class="col-md-5 p-3 login justify-content-md-center"><hr>
        <img src="https://mvhs.mvla.net/images/logo.png" alt="mvhs-logo" class="i">
        <br><br>
        <h1 class="h3 mb-3 font-weight-normal text-center">Pod Leader Login</h1>
        <br>
        <p v-if="incorrectAuth">Incorrect username or password entered - please try again</p>
        <form v-on:submit.prevent="login">
          <div class="form-group">
            <input type="text" name="username" id="user" v-model="username" class="form-control" placeholder="Username">
          </div>
          <br><br>
          <div class="form-group">
            <input type="password" name="password" id="pass" v-model="password" class="form-control" placeholder="Password">
          </div>
          <br>
          <button type="submit" class="btn btn-md btn-primary btn-block">Login</button>
        </form>
        <br><br><br>
        <hr>
      </div>
    </div>
  </div>
  <br><br></div>
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
.background{
  background-color:#FFD700;
}

.login{
  background-color:white;
  transition: 0.3s;
  

}

.i {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
  
}
.login:hover {
  box-shadow: 0 20px 20px 0 rgba(0,0,0,0.2);
}


</style>