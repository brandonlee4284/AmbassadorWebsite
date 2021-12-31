import { createStore } from 'vuex'
import { getAPI } from './axios-api'

export default createStore({
  state: {
     accessToken: null,
     refreshToken: null,
     localStorageToken: null,
     APIData: '',
     
  },
  mutations: {
    updateStorage (state, { access, refresh, localStorageToken }) {
      state.accessToken = access
      state.refreshToken = refresh
      localStorage.setItem('userDetails', localStorageToken)

    },
    destroyToken (state) {
      state.accessToken = null
      state.refreshToken = null
      localStorage.removeItem('userDetails')   
    },
    /*
    saveTokenInLocalStorage(tokenDetails){
      localStorage.setItem('userDetails', JSON.stringify(tokenDetails));
    },
    deleteTokenInLocalStorage(){
      localStorage.removeItem('userDetails');
    },
    */
   
   
  },
  getters: {
    loggedIn (state) {
      //console.log(state.accessToken != null)
      state.localStorageToken = localStorage.getItem('userDetails')
      //console.log(state.localStorageToken)
      return state.accessToken != null
    }
  },
  actions: {
    userLogout (context) {
      if (context.getters.loggedIn) {
          context.commit('destroyToken')
          //context.commit('deleteTokenInLocalStorage')
      }
    },
    userLogin (context, usercredentials) {      
        return new Promise((resolve, reject) => {   
          getAPI.post('/api-token/', {
            username: usercredentials.username,
            password: usercredentials.password
          })   
            .then(response => {
              context.commit('updateStorage', { access: response.data.access, refresh: response.data.refresh, localStorageToken: response.data.refresh })
              //context.commit('saveTokenInLocalStorage', { refresh: response.data.refresh })
              resolve()
            })
            .catch(err => {
              reject(err)
            })
        })
      
      
    }
  }
})