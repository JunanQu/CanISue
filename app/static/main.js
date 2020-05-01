(function () {

  'use strict';

  angular.module('WordcountApp', [])

    .controller('WordcountController', ['$scope', '$log', '$http', '$timeout',
      function ($scope, $log, $http, $timeout) {

        $scope.submitButtonText = 'Search';
        $scope.loading = false;
        $scope.urlerror = false;
        $scope.finished = false;
        $scope.reddit_data = []
        $scope.caselaw_show = false;

        $scope.getResults = function () {
          // get the URL from the input
          var userInput = $scope.url;
          var min_date = $scope.min_date
          var state = $scope.state
          // fire the API request
          $http.post('/start', { 'data': [userInput, min_date, state] }).
            success(function (results) {
              $log.log(results);
              getWordCount(results);
              $scope.wordcounts = null;
              $scope.loading = true;
              $scope.submitButtonText = 'Loading...';
              $scope.urlerror = false;
            }).
            error(function (error) {
              $log.log(error);
            });

        };

        function getWordCount(jobID) {

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
                  $scope.caselaw_data = data[4]
                  $scope.caselaw_show = data[4][0]
                  if ($scope.caselaw_show == -1) {
                    $scope.caselaw_show = false
                  }
                  $scope.loading = false;
                  $scope.submitButtonText = "Search";
                  $scope.wordcounts = data;
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

    .directive('wordCountChart', ['$parse', function ($parse) {
      return {
        restrict: 'E',
        replace: true,
        template: '<div id="chart"></div>',
        link: function (scope) {
          scope.$watch('wordcounts', function () {
            d3.select('#chart').selectAll('*').remove();
            var data = scope.wordcounts;
            for (var word in data) {
              var key = data[word][0];
              var value = data[word][1];
              d3.select('#chart')
                .append('div')
                .selectAll('div')
                .data(word)
                .enter()
                .append('div')
                .style('width', function () {
                  return (value * 3) + 'px';
                })
                .text(function (d) {
                  return key;
                });
            }
          }, true);
        }
      };
    }]);

}());
