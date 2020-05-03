(function () {

  'use strict';

  angular.module('canisueApp', [])

    .controller('appController', ['$scope', '$log', '$http', '$timeout',
      function ($scope, $log, $http, $timeout) {

        $scope.submitButtonText = 'Search';
        $scope.loading = false;
        $scope.urlerror = false;
        $scope.finished = false;
        $scope.reddit_data = []
        $scope.caselaw_show = false;
        $scope.disclaimer_show = true;
        $scope.debug_message = ''
        $scope.output_message = ''
        $scope.caselaw_message = ''

        $scope.getResults = function () {
          // get the input
          var userInput = $scope.url;
          var min_date = $scope.min_date
          var state = $scope.state
          // fire the API request
          $http.post('/start', { 'data': [userInput, min_date, state] }).
            success(function (results) {
              $log.log(results);
              collect_job(results);
              $scope.queried_data = null;
              $scope.loading = true;
              $scope.finished = false;
              $scope.submitButtonText = 'Loading...';
              $scope.urlerror = false;
              $scope.disclaimer_show = false;
            }).
            error(function (error) {
              $log.log(error);
            });

        };

        function collect_job(jobID) {

          var timeout = '';

          var poller = function () {
            // fire another request
            $http.get('/results/' + jobID).
              success(function (data, status, headers, config) {
                if (status === 202) {
                  $log.log(data, status);
                } else if (status === 200) {
                  $log.log(data);
                  $scope.reddit_data = data[3]

                  if (data === undefined || data[4] === undefined) {
                    $scope.caselaw_show = -1
                    $scope.caselaw_data = []
                  } else {
                    $scope.caselaw_show = data[4][0]
                    $scope.caselaw_data = data[4]
                  }

                  $scope.output_message = data[2]
                  $scope.caselaw_message = data[5]
                  $scope.debug_message = data[7]
                  if ($scope.caselaw_show == -1) {
                    $scope.caselaw_show = false
                    $scope.urlerror = true
                  }

                  $scope.loading = false;
                  $scope.submitButtonText = "Search";
                  $scope.queried_data = data;
                  $timeout.cancel(timeout);
                  $scope.finished = true;
                  return false;
                }
                // continue to call the poller() function every 2 seconds
                // until the timeout is cancelled
                timeout = $timeout(poller, 2000);
              }).
              error(function (error) {
                $log.log(error);
                $scope.loading = false;
                $scope.submitButtonText = "Search";
                $scope.urlerror = true;
              });
          };

          poller();

        }

      }])


}());
